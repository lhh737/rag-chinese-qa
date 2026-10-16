"""MCP server 测试：离线 stub 管道 + SDK 内存传输（不起子进程、不触网、不需索引）。"""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("mcp")
from mcp import Client  # noqa: E402

from ragqa.mcp_server import create_mcp_server  # noqa: E402
from ragqa.tools import EXCERPT_CHARS  # noqa: E402
from ragqa.types import (  # noqa: E402
    Citation,
    ContextBlock,
    QueryResult,
    RetrievalResult,
    RetrievalTrace,
)

EXPECTED_TOOLS = {"search_documents", "get_document_content", "list_documents", "answer"}


class _FakeRetriever:
    """鸭子类型：只实现 tools.search_documents 用到的那一个方法。"""

    def retrieve(self, query: str) -> RetrievalResult:
        return RetrievalResult(
            blocks=[
                ContextBlock(rank=1, parent_id="a:c1", doc_id="a", source="a.pdf",
                             text="答案 42", rerank_score=0.91),
                ContextBlock(rank=2, parent_id="b:c1", doc_id="b", source="b.pdf",
                             text="x" * (EXCERPT_CHARS + 50), rerank_score=0.42),
            ],
            trace=RetrievalTrace(latency_ms=40.0),
        )


class FakePipeline:
    """与 tests/test_api.py 同构的门面假件。"""

    def __init__(self) -> None:
        self.retriever = _FakeRetriever()

    def get_file_list(self) -> list[dict]:
        return [{"doc_id": "a", "source": "a.pdf"}]

    def get_document_content(self, doc_id: str) -> str | None:
        return "文档全文" if doc_id == "a" else None

    def query(self, question: str, prompt_version: str | None = None) -> QueryResult:
        return QueryResult(
            answer="根据参考资料，答案是 42[1]。",
            contexts=[ContextBlock(rank=1, parent_id="a:c1", doc_id="a", source="a.pdf", text="答案 42")],
            citations=[Citation(answer_sentence="答案是 42[1]。", marker=1, valid=True, source_doc="a")],
            trace={"config_id": "test-config", "totals": {"latency_ms": {"total": 123.4}}},
        )


def _call(tool: str, args: dict | None = None):
    """进程内跑一次工具调用，返回 (工具名集合, CallToolResult)。"""

    async def go():
        async with Client(create_mcp_server(FakePipeline())) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            return names, await client.call_tool(tool, args or {})

    return asyncio.run(go())


def test_server_exposes_expected_tools() -> None:
    names, _ = _call("list_documents")
    assert EXPECTED_TOOLS <= names


def test_search_documents_returns_ranked_blocks() -> None:
    _, res = _call("search_documents", {"query": "答案是多少"})
    assert res.is_error is False
    payload = res.structured_content
    # 必须返回 JSON 对象：结构化结果在 2025-06-18 / 2025-11-25 协议下仅允许 object
    assert isinstance(payload, dict)
    assert payload["n_blocks"] == 2
    first = payload["blocks"][0]
    assert first["rank"] == 1 and first["doc_id"] == "a" and first["source"] == "a.pdf"
    assert first["rerank_score"] == 0.91


def test_search_documents_truncates_long_block() -> None:
    _, res = _call("search_documents", {"query": "q"})
    second = res.structured_content["blocks"][1]
    assert len(second["text"]) == EXCERPT_CHARS + 1  # 截断 + 省略号
    assert second["text"].endswith("…")


def test_search_documents_limit_argument() -> None:
    _, res = _call("search_documents", {"query": "q", "limit": 1})
    assert res.structured_content["n_blocks"] == 1


def test_get_document_content_found_and_missing() -> None:
    _, found = _call("get_document_content", {"doc_id": "a"})
    assert found.structured_content == {"doc_id": "a", "found": True, "content": "文档全文"}

    _, missing = _call("get_document_content", {"doc_id": "不存在"})
    assert missing.structured_content == {"doc_id": "不存在", "found": False, "content": ""}


def test_answer_carries_citations_and_config_id() -> None:
    _, res = _call("answer", {"question": "答案是多少"})
    payload = res.structured_content
    assert payload["answer"].endswith("[1]。")
    assert payload["citations"] == [{"marker": 1, "valid": True, "source_doc": "a"}]
    assert payload["n_contexts"] == 1
    assert payload["config_id"] == "test-config"


def test_tools_never_write_to_stdout(capsys) -> None:
    """stdio 传输下 stdout 是协议线：工具路径不得 print（诊断只能走 logger→stderr）。"""
    from ragqa import tools

    pipeline = FakePipeline()
    tools.search_documents(pipeline, "q")
    tools.get_document_content(pipeline, "a")
    tools.list_documents(pipeline)
    tools.answer(pipeline, "q")
    assert capsys.readouterr().out == ""
