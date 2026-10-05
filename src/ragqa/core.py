"""
统一 RAG 管线：支持 HyDE + 混合检索 + 简单检索三种模式。
"""
from __future__ import annotations

import os

from config.settings import UPLOAD_DIR
from rag.document_loader import get_file_info, load_and_split, load_and_split_parent_child
from rag.generator import generate
from rag.hybrid_retriever import HybridRetriever
from rag.hyde import generate_hypothetical_doc
from rag.vector_store import VectorStore

SYSTEM_PROMPT = "你是一个专业的中文文档问答助手。请根据提供的文档内容，准确、简洁地回答用户的问题。如果文档中没有相关信息，请如实告知。"


class RAGPipeline:
    def __init__(self, use_hybrid: bool = True, use_hyde: bool = True):
        """
        Args:
            use_hybrid: True 使用混合检索（BM25+向量+重排序+父子分块）。
            use_hyde:   True 使用 HyDE — 先生成假设性答案再检索，可提升召回率。
        """
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        self.vector_store = VectorStore()
        self.hybrid = HybridRetriever(self.vector_store) if use_hybrid else None
        self.use_hybrid = use_hybrid
        self.use_hyde = use_hyde and use_hybrid  # HyDE 依赖混合检索

    # ── 文档管理 ──────────────────────────────────────────

    def upload_and_index(self, filepath: str) -> tuple[bool, str]:
        if not os.path.isfile(filepath):
            return False, f"文件不存在: {filepath}"

        ext = os.path.splitext(filepath)[1].lower()
        if ext not in (".txt", ".pdf"):
            return False, f"不支持的文件格式: {ext}，仅支持 .txt / .pdf"

        try:
            info = get_file_info(filepath)

            if self.use_hybrid:
                parent_docs, child_docs = load_and_split_parent_child(filepath)
                if not child_docs:
                    return False, "文档解析失败，未能提取到文本内容"
                ok = self.vector_store.add_parent_child_documents(parent_docs, child_docs, info)
            else:
                chunks = load_and_split(filepath)
                if not chunks:
                    return False, "文档解析失败，未能提取到文本内容"
                ok = self.vector_store.add_documents(chunks, info)

            if not ok:
                return False, f"文档已存在: {info['filename']}"
            chunk_count = len(child_docs) if self.use_hybrid else len(chunks)  # type: ignore
            return True, f"已索引 {chunk_count} 个文本块"
        except Exception as e:
            return False, f"处理失败: {str(e)}"

    # ── 检索核心 ──────────────────────────────────────────

    def _retrieve(self, question: str) -> tuple[str, list[dict]]:
        """
        根据配置模式执行检索，返回 (context, provenance)。
        """
        if not self.hybrid:
            # 简单向量检索
            docs = self.vector_store.search(question)
            if not docs:
                return "", []
            context = "\n\n".join(d.page_content for d in docs)
            provenance = [
                {"source": d.metadata.get("source", ""), "excerpt": d.page_content[:200]}
                for d in docs
            ]
            return context, provenance

        # ---- HyDE：先用 LLM 生成假设性答案作为检索 query ----
        if self.use_hyde:
            hyde_query = generate_hypothetical_doc(question)
        else:
            hyde_query = question

        # 混合检索（BM25 + 向量 + 重排序）
        return self.hybrid.retrieve_for_rag(hyde_query)

    # ── 问答 ──────────────────────────────────────────────

    def query(self, question: str) -> str:
        if self.vector_store.is_empty:
            return "知识库为空，请先上传文档。"

        context, provenance = self._retrieve(question)
        if not context.strip():
            return "未找到相关文档内容。"

        answer = generate(SYSTEM_PROMPT, context=context, question=question)

        # 附加溯源信息
        if provenance:
            lines = [answer, "\n--- 参考来源 ---"]
            seen = set()
            for s in provenance:
                src = s.get("source", "")
                if src and src not in seen:
                    seen.add(src)
                    lines.append(f"- {src}")
            return "\n".join(lines)
        return answer

    def query_with_sources(self, question: str) -> dict:
        """返回 {answer, sources} 结构，供 UI 展示详细溯源。"""
        if self.vector_store.is_empty:
            return {"answer": "知识库为空，请先上传文档。", "sources": []}

        context, provenance = self._retrieve(question)
        if not context.strip():
            return {"answer": "未找到相关文档内容。", "sources": []}

        answer = generate(SYSTEM_PROMPT, context=context, question=question)
        return {"answer": answer, "sources": provenance}

    # ── 文件列表 ──────────────────────────────────────────

    def get_file_list(self) -> list[dict]:
        return self.vector_store.file_list

    def get_document_content(self, md5: str) -> str | None:
        return self.vector_store.get_document_content(md5)
