"""MCP server（P9）：把 RAG 能力暴露为标准 MCP 工具，供任意 MCP 客户端调用。

依赖 mcp SDK v2（optional extra `mcp`，`uv sync --extra mcp`）——核心链路不依赖它。

规范口径（2026-10 实测）：MCP 规范自 2026-07-28 起为破坏性重构，SDK v2 对应实现：
`mcp.server.MCPServer`（v1 的 `FastMCP` 已移除）、`initialize` 握手由 `server/discover`
取代、会话无状态。本模块只用 v2 公开 API（构造 + `@server.tool()` + `run()`）。

**stdio 传输硬约束**：stdout 是协议线，只允许 MCP 协议消息。本模块**禁止 print 到 stdout**，
所有诊断走 `ragqa.utils.logging.logger`（其 StreamHandler 默认写 stderr）。

运行：
    uv run ragqa mcp                          # stdio（默认），供本地 MCP 客户端拉起
    uv run ragqa mcp --transport http --port 8000   # Streamable HTTP → /mcp
"""
from __future__ import annotations

from typing import Any

from ragqa import tools
from ragqa.core import RAGPipeline
from ragqa.utils.logging import logger


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("rag-chinese-qa")
    except Exception:  # noqa: BLE001 — 未安装为包时（如直接跑源码）不必失败
        return ""


def create_mcp_server(pipeline: RAGPipeline | None = None):
    """构造 MCP server；`pipeline` 可注入（测试用假件），默认首次调用时惰性构建。

    与 `ragqa.api.create_app` 同一注入约定：CLI / Web UI / FastAPI / 评测引擎 / MCP
    共用同一个 `RAGPipeline` 门面，故工具返回与问答主链路完全同源。
    """
    from mcp.server import MCPServer  # 懒导入：未装 mcp extra 时不影响其余功能

    server = MCPServer(
        name="rag-chinese-qa",
        title="中文 RAG 文档问答",
        instructions=(
            "对已索引的中文论文/文档库做检索与问答。"
            "先用 search_documents 找相关块，再按需 get_document_content 取原文；"
            "需要成句答案时用 answer（带 [n] 行内引用）。"
        ),
        version=_package_version(),
    )

    holder: dict[str, RAGPipeline] = {}
    if pipeline is not None:
        holder["p"] = pipeline

    def _pipeline() -> RAGPipeline:
        if "p" not in holder:
            from ragqa.retrieval.store import VectorStore

            holder["p"] = RAGPipeline(store=VectorStore())
            logger.info("[mcp] 管道已就绪")
        return holder["p"]

    # 注意：所有工具一律返回 JSON **对象**（不用数组）——SDK 类型定义载明
    # structured_content 在 2025-06-18 / 2025-11-25 协议下仅允许 object，
    # 只有 2026-07-28 起才放开为任意 JSON 值。包成对象对新旧客户端都安全。

    @server.tool(name="search_documents")
    def search_documents(query: str, limit: int | None = None) -> dict[str, Any]:
        """检索知识库，返回与查询最相关的文档块（含来源与重排分数）。不生成答案。"""
        return tools.search_documents(_pipeline(), query, limit)

    @server.tool(name="get_document_content")
    def get_document_content(doc_id: str) -> dict[str, Any]:
        """取某篇文档的原文全文。doc_id 可从 search_documents 或 list_documents 的结果获得。"""
        return tools.get_document_content(_pipeline(), doc_id)

    @server.tool(name="list_documents")
    def list_documents() -> dict[str, Any]:
        """列出知识库中已索引的文档。"""
        return tools.list_documents(_pipeline())

    @server.tool(name="answer")
    def answer(question: str) -> dict[str, Any]:
        """对问题生成带 [n] 行内引用的答案（库外问题会明确拒答）。"""
        return tools.answer(_pipeline(), question)

    return server


def main(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8000) -> int:
    """CLI 入口（`ragqa mcp`）。注意：stdio 下不得向 stdout 写入任何非协议内容。"""
    if transport == "http":
        transport = "streamable-http"
    server = create_mcp_server()
    logger.info("[mcp] 启动：transport=%s%s", transport, f" {host}:{port}" if transport != "stdio" else "")
    if transport == "stdio":
        server.run("stdio")
    else:
        server.run(transport, host=host, port=port)  # type: ignore[arg-type]
    return 0
