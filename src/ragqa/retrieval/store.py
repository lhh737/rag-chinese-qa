"""
向量存储：FAISS 向量库 + 父子分块 + BM25 语料 + 文件清单。
同时兼容简单 RAGPipeline 和混合检索 HybridRetriever。
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any

import jieba
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config.settings import FAISS_PERSIST_DIR, RETRIEVAL_K
from model.factory import get_embed_model
from utils.config_handler import faiss_conf
from utils.logger_handler import logger


def _zh_tokenize(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    try:
        return [t for t in jieba.cut(text) if t.strip()]
    except Exception:
        return list(text)


class VectorStore:
    """统一向量存储：FAISS 子块检索 + 父块上下文还原 + BM25 混合检索 + 文件清单。"""

    def __init__(self):
        self.persist_dir = FAISS_PERSIST_DIR
        self.index_path = os.path.join(self.persist_dir, "faiss_index")
        self.manifest_path = os.path.join(self.persist_dir, "manifest.json")
        self.parent_store_path = os.path.join(self.persist_dir, "parent_store.json")
        os.makedirs(self.persist_dir, exist_ok=True)

        # FAISS store（子块索引）
        self.store: FAISS | None = None
        # 子块列表（用于 BM25 检索 + 混合检索遍历）
        self.child_documents: list[Document] = []
        # 子块对应的分词结果（BM25 用）
        self.tokenized_corpus: list[list[str]] = []
        # 父块存储：parent_id -> {"page_content": str, "metadata": dict}
        self.parent_store: dict[str, dict[str, Any]] = {}
        # 文件清单：md5 -> 文件信息
        self.manifest: dict[str, dict] = {}

        self._load()

    # ── 持久化 ──────────────────────────────────────────────

    def _load(self):
        """从磁盘恢复全部状态。"""
        # FAISS
        if os.path.isdir(self.index_path):
            try:
                self.store = FAISS.load_local(
                    self.index_path,
                    get_embed_model(),
                    index_name="index",
                    allow_dangerous_deserialization=True,
                )
                # 验证维度一致（切换嵌入模型后旧索引会不兼容）
                if self.store.index:
                    try:
                        expected = len(get_embed_model().embed_query("test"))
                        if self.store.index.d != expected:
                            logger.info("[VectorStore] FAISS 维度不匹配（旧 %d，新 %d），重建",
                                        self.store.index.d, expected)
                            self.store = None
                    except Exception as e:
                        logger.warning("[VectorStore] 无法检测嵌入维度: %s", e)
            except Exception as e:
                logger.warning("[VectorStore] FAISS 加载失败，将重建: %s", e)
                self.store = None

        # 子块文档（配合 FAISS 索引重建用；FAISS 本身不存 doc 内容外的 metadata）
        child_path = os.path.join(self.persist_dir, "child_docs.json")
        if os.path.isfile(child_path):
            try:
                with open(child_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self.child_documents = [Document(**d) if isinstance(d, dict) else d for d in raw]
                self.tokenized_corpus = [_zh_tokenize(d.page_content) for d in self.child_documents]
            except Exception as e:
                logger.warning("[VectorStore] 子块文档恢复失败: %s", e)
                self.child_documents = []
                self.tokenized_corpus = []

        # 父块存储
        if os.path.isfile(self.parent_store_path):
            try:
                with open(self.parent_store_path, "r", encoding="utf-8") as f:
                    self.parent_store = json.load(f)
            except Exception as e:
                logger.warning("[VectorStore] 父块存储恢复失败: %s", e)
                self.parent_store = {}

        # 文件清单
        if os.path.isfile(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self.manifest = json.load(f)
            except Exception as e:
                logger.warning("[VectorStore] 文件清单恢复失败: %s", e)
                self.manifest = {}

    def _save(self):
        """保存全部状态到磁盘。"""
        # FAISS
        if self.store:
            self.store.save_local(self.index_path, index_name="index")

        # 子块文档
        child_path = os.path.join(self.persist_dir, "child_docs.json")
        try:
            with open(child_path, "w", encoding="utf-8") as f:
                json.dump(
                    [{"page_content": d.page_content, "metadata": d.metadata} for d in self.child_documents],
                    f, ensure_ascii=False, indent=2,
                )
        except Exception as e:
            logger.error("[VectorStore] 保存子块文档失败: %s", e)

        # 父块存储
        try:
            with open(self.parent_store_path, "w", encoding="utf-8") as f:
                json.dump(self.parent_store, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("[VectorStore] 保存父块存储失败: %s", e)

        # 文件清单
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(self.manifest, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("[VectorStore] 保存文件清单失败: %s", e)

    # ── 属性 ──────────────────────────────────────────────

    @property
    def faiss_store(self) -> FAISS | None:
        """hybrid_retriever 兼容属性。"""
        return self.store

    @property
    def is_empty(self) -> bool:
        return self.store is None or len(self.child_documents) == 0

    @property
    def file_list(self) -> list[dict]:
        return sorted(self.manifest.values(), key=lambda x: x.get("loaded_at", ""), reverse=True)

    # ── 文档管理 ──────────────────────────────────────────

    def add_documents(self, chunks: list[Document], file_info: dict) -> bool:
        """简单模式：添加普通分块（无父子分块）。"""
        md5 = file_info["md5"]
        if md5 in self.manifest:
            return False

        # 给每个 chunk 生成唯一 ID
        for d in chunks:
            if "chunk_uid" not in d.metadata:
                d.metadata["chunk_uid"] = str(uuid.uuid4())

        if self.store is None:
            self.store = FAISS.from_documents(chunks, get_embed_model())
        else:
            self.store.add_documents(chunks)

        self.child_documents.extend(chunks)
        self.tokenized_corpus.extend(_zh_tokenize(d.page_content) for d in chunks)

        self.manifest[md5] = file_info
        self._save()
        return True

    def add_parent_child_documents(
        self,
        parent_docs: list[Document],
        child_docs: list[Document],
        file_info: dict,
    ) -> bool:
        """混合检索模式：添加父子分块。"""
        md5 = file_info["md5"]
        if md5 in self.manifest:
            return False

        # 建立父块索引：使用调用方已设置的 parent_id（由 load_and_split_parent_child 生成）
        for i, parent in enumerate(parent_docs):
            pid = parent.metadata.get("parent_id") or str(uuid.uuid4())
            parent.metadata["parent_id"] = pid
            self.parent_store[pid] = {
                "page_content": parent.page_content,
                "metadata": {k: v for k, v in parent.metadata.items() if k != "parent_id"},
            }

        # 子块链接到父块（子块应已从 load_and_split_parent_child 获得匹配的 parent_id）
        for child in child_docs:
            if "chunk_uid" not in child.metadata:
                child.metadata["chunk_uid"] = str(uuid.uuid4())

        if self.store is None:
            self.store = FAISS.from_documents(child_docs, get_embed_model())
        else:
            self.store.add_documents(child_docs)

        self.child_documents.extend(child_docs)
        self.tokenized_corpus.extend(_zh_tokenize(d.page_content) for d in child_docs)

        self.manifest[md5] = file_info
        self._save()
        return True

    def search(self, query: str, k: int = None) -> list[Document]:
        """简单向量检索。"""
        if self.store is None:
            return []
        k = k or RETRIEVAL_K
        return self.store.similarity_search(query, k=k)

    def get_document_content(self, md5: str) -> str | None:
        """根据 MD5 获取原始文件内容。"""
        info = self.manifest.get(md5)
        if not info:
            return None
        path = info.get("path", "")
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return None

    def get_file_info_by_md5(self, md5: str) -> dict | None:
        return self.manifest.get(md5)

    def clear(self):
        """清空所有数据。"""
        self.store = None
        self.child_documents = []
        self.tokenized_corpus = []
        self.parent_store = {}
        self.manifest = {}
        # 清理磁盘文件
        import shutil
        if os.path.isdir(self.index_path):
            shutil.rmtree(self.index_path, ignore_errors=True)


# ===== 向后兼容别名（供 rag_service / hybrid_retriever 旧引用使用） =====
VectorStoreService = VectorStore
