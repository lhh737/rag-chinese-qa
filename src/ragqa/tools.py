"""工具层：对 `RAGPipeline` 的薄封装，供两种消费方式共用（P9）。

- 对外：MCP server（`ragqa.mcp_server`）把它们注册为 MCP tools；
- 对内：工具调用变体（`ragqa.agent`）把它们转成 OpenAI function-calling schema。

设计约束：本模块**不依赖任何 MCP / OpenAI 协议细节**，只做「输入 → 调管道 → 纯 dict」，
两种协议各自在最外层适配。检索逻辑一律复用 `HybridRetriever.retrieve`，不另写。
"""
from __future__ import annotations

from typing import Any

from ragqa.core import RAGPipeline

# 单个上下文块回传给模型的最大字符数（防止长父块撑爆调用方上下文窗口）
EXCERPT_CHARS = 800


def _excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def search_documents(pipeline: RAGPipeline, query: str, limit: int | None = None) -> dict[str, Any]:
    """检索文档块（不生成答案）。

    复用 `HybridRetriever.retrieve`——与问答主链路同一条管线、同一份配置，
    故返回块的 rank / 分数与 `[n]` 引用编号口径一致。

    Args:
        pipeline: 统一门面。
        query: 检索查询。
        limit: 最多返回几个块（默认全部，即该配置下动态 TopK 的结果）。
    """
    result = pipeline.retriever.retrieve(query)
    blocks = result.blocks[:limit] if limit else result.blocks
    return {
        "query": query,
        "n_blocks": len(blocks),
        "blocks": [
            {
                "rank": b.rank,
                "doc_id": b.doc_id,
                "source": b.source,
                "rerank_score": b.rerank_score,
                "text": _excerpt(b.text),
            }
            for b in blocks
        ],
    }


def get_document_content(pipeline: RAGPipeline, doc_id: str) -> dict[str, Any]:
    """取某篇文档的原文全文。doc_id 可从 search_documents / list_documents 的结果获得。"""
    content = pipeline.get_document_content(doc_id)
    if content is None:
        return {"doc_id": doc_id, "found": False, "content": ""}
    return {"doc_id": doc_id, "found": True, "content": content}


def list_documents(pipeline: RAGPipeline) -> dict[str, Any]:
    """列出知识库中已索引的文档。"""
    files = pipeline.get_file_list()
    return {"n_documents": len(files), "documents": files}


def answer(pipeline: RAGPipeline, question: str) -> dict[str, Any]:
    """走完整问答链路：检索 → 生成，返回带 `[n]` 行内引用的答案。

    与 CLI / Web UI / 评测引擎共用同一条 `RAGPipeline.query`。
    """
    res = pipeline.query(question)
    return {
        "answer": res.answer,
        "citations": [
            {"marker": c.marker, "valid": c.valid, "source_doc": c.source_doc}
            for c in res.citations
        ],
        "n_contexts": len(res.contexts),
        "config_id": res.trace.get("config_id", ""),
        "latency_ms": res.trace.get("totals", {}).get("latency_ms", {}).get("total"),
    }


# 工具调用变体（ragqa.agent）用的 OpenAI function-calling schema；
# MCP 侧不用它——MCP 由类型注解 + docstring 自动生成 schema。
FUNCTION_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "检索知识库，返回与查询最相关的文档块（含来源与重排分数）。不生成答案。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索查询"},
                    "limit": {"type": "integer", "description": "最多返回几个块，默认全部"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_content",
            "description": "取某篇文档的原文全文（doc_id 来自检索结果）。",
            "parameters": {
                "type": "object",
                "properties": {"doc_id": {"type": "string", "description": "文档 ID"}},
                "required": ["doc_id"],
            },
        },
    },
]
