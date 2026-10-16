"""单轮工具选择变体测试（离线假件，不触网、不需索引）。

覆盖：不调用工具 / 单查询 / 多查询合并 / 参数非法 JSON / 工具选择失败回退 / trace 契约。
"""
from __future__ import annotations

import json

import pytest

from ragqa.agentic import ToolCallingPipeline
from ragqa.config.loader import load_retrieval_config
from ragqa.core import RAGPipeline
from ragqa.evaluation.runner import mrr, recall_at_k
from ragqa.retrieval.hybrid import format_context
from ragqa.types import ChunkHit, ContextBlock, RetrievalResult, RetrievalTrace

_DOC_TEXT = {"d1": "文档一正文", "d2": "文档二正文"}

REQUIRED_TRACE_KEYS = [
    "schema_version", "trace_id", "timestamp", "question", "config_id",
    "hyde", "vector", "bm25", "rrf", "rerank", "context", "generation", "totals", "versions",
]


class _FakeStore:
    is_empty = False
    meta = {"count": 2, "dim": 8, "embed_model": "fake-embed", "pipeline_version": "test"}


class _FakeRetriever:
    """按查询串里出现的 doc_id 决定返回哪些块——用于区分单查询 / 多查询。"""

    def __init__(self, fail_on: set[str] | None = None) -> None:
        self.queries: list[str] = []
        self._fail_on = fail_on or set()

    def retrieve(self, query: str) -> RetrievalResult:
        self.queries.append(query)
        if query in self._fail_on:
            raise RuntimeError(f"stub 检索失败: {query}")
        docs = [d for d in ("d1", "d2") if d in query] or ["d1"]
        blocks = [
            ContextBlock(rank=i + 1, parent_id=f"{d}:p1", doc_id=d, source=f"{d}.pdf",
                         text=_DOC_TEXT[d], rerank_score=round(0.9 - i * 0.1, 6))
            for i, d in enumerate(docs)
        ]
        trace = RetrievalTrace(
            vector_hits=[ChunkHit(f"{d}:c1", 0.9, i + 1, f"{d}:p1", d) for i, d in enumerate(docs)],
            bm25_hits=[ChunkHit(f"{d}:c1", 3.0, i + 1, f"{d}:p1", d) for i, d in enumerate(docs)],
            fusion={"strategy": "rrf", "k": 60, "n_vector": len(docs), "n_bm25": len(docs),
                    "n_fused": len(docs),
                    "fused": [{"chunk_uid": f"{d}:c1", "rrf_score": round(0.5 - i * 0.01, 6)}
                              for i, d in enumerate(docs)]},
            rerank={"scores": [0.9 - i * 0.1 for i in range(len(docs))],
                    "top": [f"{d}:c1" for d in docs], "latency_ms": 5.0,
                    "usage": {"total_tokens": 10}, "error": None},
            context_parent_ids=[b.parent_id for b in blocks],
            context_chars=len(_DOC_TEXT["d1"]),
            latency_ms=20.0,
        )
        return RetrievalResult(blocks=blocks, trace=trace, context=format_context(blocks))


# ── 假生成客户端：带 tools= 的调用返回工具选择，否则返回答案 ──

class _Usage:
    prompt_tokens = 30
    completion_tokens = 5
    total_tokens = 35


class _Fn:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, name: str, arguments: str) -> None:
        self.function = _Fn(name, arguments)


class _Msg:
    def __init__(self, content: str, tool_calls: list[_ToolCall] | None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class _Resp:
    def __init__(self, msg: _Msg) -> None:
        self.choices = [type("_C", (), {"message": msg})()]
        self.usage = _Usage()


class ToolSelectClient:
    """工具选择轮返回给定的 tool_calls；生成轮返回固定答案。"""

    def __init__(self, tool_calls: list[tuple[str, object]] | None = None,
                 raise_on_select: bool = False) -> None:
        self.tool_calls = tool_calls or []
        self.raise_on_select = raise_on_select
        self.calls: list[dict] = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "tools" in kwargs:
            if self.raise_on_select:
                raise RuntimeError("stub 工具选择失败")
            tcs = [
                _ToolCall(name, arg if isinstance(arg, str) else json.dumps(arg))
                for name, arg in self.tool_calls
            ]
            return _Resp(_Msg("", tcs))
        return _Resp(_Msg("根据参考资料，答案是 [1]。", None))


def _make(tool_calls=None, *, raise_on_select=False, fail_on=None):
    client = ToolSelectClient(tool_calls, raise_on_select=raise_on_select)
    base = RAGPipeline(store=_FakeStore(), retrieval_config=load_retrieval_config(), gen_client=client)
    retriever = _FakeRetriever(fail_on=fail_on)
    base.retriever = retriever
    return ToolCallingPipeline(base), client, retriever


def _search(q: str):
    return ("search_documents", {"query": q})


# ── 用例 ─────────────────────────────────────────────

def test_no_tool_call_yields_zero_hit_path():
    pipe, _, retriever = _make([])
    res = pipe.query("问题")
    assert res.answer == "未找到相关文档内容。"
    assert retriever.queries == []
    assert res.trace["agentic"]["n_tool_calls"] == 0


def test_single_search_uses_model_query():
    pipe, _, retriever = _make([_search("找 d1 的资料")])
    res = pipe.query("问题")
    assert retriever.queries == ["找 d1 的资料"]  # 用的是模型给出的查询，不是原问题
    assert res.answer.endswith("[1]。")
    assert res.trace["agentic"]["sub_queries"] == ["找 d1 的资料"]
    assert res.trace["context"]["sources"] == ["d1.pdf"]


def test_multi_query_merges_and_keeps_both_docs():
    pipe, _, retriever = _make([_search("d1 的资料"), _search("d2 的资料")])
    res = pipe.query("比较 d1 和 d2")
    assert retriever.queries == ["d1 的资料", "d2 的资料"]
    trace = res.trace
    assert trace["rrf"]["strategy"] == "rrf_multiquery"
    assert trace["rrf"]["n_subqueries"] == 2
    # 合并后两个来源都进了上下文（固定管线单次检索做不到这一点）
    assert set(trace["context"]["sources"]) == {"d1.pdf", "d2.pdf"}
    # rank 重编号连续，供 [n] 引用解析
    assert [b.rank for b in res.contexts] == [1, 2]
    # 子块层融合包含两个子查询的 chunk_uid
    uids = {f["chunk_uid"] for f in trace["rrf"]["fused"]}
    assert {"d1:c1", "d2:c1"} <= uids


def test_merged_trace_is_comparable_by_evaluator_metrics():
    """合并后的 trace 必须能被评测器的 recall@5 / MRR 直接消费。"""
    pipe, _, _ = _make([_search("d1 的资料"), _search("d2 的资料")])
    trace = pipe.query("比较 d1 和 d2").trace
    assert recall_at_k(trace, ["d1", "d2"], 5) == 1.0
    assert mrr(trace, ["d1", "d2"]) == 1.0


def test_invalid_json_arguments_are_dropped():
    pipe, _, retriever = _make([("search_documents", "{不是合法 json")])
    res = pipe.query("问题")
    assert retriever.queries == []
    assert res.answer == "未找到相关文档内容。"


def test_unknown_tool_name_is_ignored():
    pipe, _, retriever = _make([("get_document_content", {"doc_id": "d1"})])
    res = pipe.query("问题")
    assert retriever.queries == []
    assert res.trace["agentic"]["n_tool_calls"] == 1  # 调用被记录，但未执行


def test_tool_select_failure_falls_back_to_zero_hit_with_error_recorded():
    pipe, _, _ = _make(raise_on_select=True)
    res = pipe.query("问题")
    assert res.answer == "未找到相关文档内容。"
    assert "stub 工具选择失败" in res.trace["agentic"]["tool_select_error"]


def test_subquery_failure_does_not_block_the_others():
    pipe, _, _ = _make([_search("d1 的资料"), _search("会失败")], fail_on={"会失败"})
    res = pipe.query("问题")
    assert res.trace["agentic"]["sub_queries"] == ["d1 的资料", "会失败"]
    assert res.trace["context"]["sources"] == ["d1.pdf"]  # 失败的那个被跳过，其余照常


def test_tool_calls_are_capped():
    pipe, _, retriever = _make([_search(f"d1 第{i}次") for i in range(6)])
    pipe.max_tool_calls = 2
    pipe.query("问题")
    assert len(retriever.queries) == 2


def test_trace_contract_and_cost_accounting():
    pipe, _, _ = _make([_search("d1 的资料")])
    trace = pipe.query("问题").trace
    for key in REQUIRED_TRACE_KEYS:
        assert key in trace, key
    assert trace["schema_version"] == 1
    assert trace["hyde"] is None                      # 本变体不使用 HyDE
    assert trace["totals"]["cost_total_cny"] > 0
    # 工具选择轮的成本必须计入，否则 agentic 成本被低估
    assert trace["totals"]["cost_by_stage_cny"]["tool_select"] > 0
    assert trace["totals"]["tokens"]["tool_select"] == 35
    # config_id 可区分于固定管线
    assert trace["config_id"].endswith("|toolsel-v1")
    json.dumps(trace, ensure_ascii=False)             # 必须可序列化


def test_generation_client_shared_with_base():
    """注入基管线的客户端应被工具选择轮复用——避免测试/离线场景漏到真实 API。"""
    client = ToolSelectClient([_search("d1 的资料")])
    base = RAGPipeline(store=_FakeStore(), retrieval_config=load_retrieval_config(), gen_client=client)
    base.retriever = _FakeRetriever()
    pipe = ToolCallingPipeline(base)  # 不显式注入
    pipe.query("问题")
    assert len(client.calls) == 2     # 工具选择轮 + 生成轮
    assert "tools" in client.calls[0] and "tools" not in client.calls[1]
