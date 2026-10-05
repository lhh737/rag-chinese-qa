"""Trace Schema v1 测试：字段完整性、可序列化、JSONL 落盘、峰谷时段。"""
from __future__ import annotations

import json
from datetime import datetime

from ragqa.trace import TraceWriter, build_trace, is_peak_time
from ragqa.types import (
    ChunkHit,
    Citation,
    ContextBlock,
    GenerationResult,
    HydeResult,
    RetrievalResult,
    RetrievalTrace,
)


def _fixture_inputs():
    hyde = HydeResult(
        enabled=True, used=True, query="hyde 输出",
        usage={"prompt_tokens": 60, "completion_tokens": 40, "total_tokens": 100}, latency_ms=10.0,
    )
    blocks = [ContextBlock(rank=1, parent_id="d1:p0000", doc_id="d1", source="d1.pdf", text="上下文内容", rerank_score=0.9)]
    rt = RetrievalTrace(
        vector_hits=[ChunkHit("d1:c00000", 0.5, 1, "d1:p0000", "d1")],
        bm25_hits=[ChunkHit("d1:c00000", 3.2, 1, "d1:p0000", "d1")],
        fusion={"strategy": "rrf", "k": 60, "fused": [{"chunk_uid": "d1:c00000", "rrf_score": 0.03}]},
        rerank={"scores": [0.9], "top": ["d1:c00000"], "latency_ms": 5.0, "usage": {"total_tokens": 20}, "error": None},
        context_parent_ids=["d1:p0000"], context_chars=len("【参考资料1】\n上下文内容"), latency_ms=20.0,
    )
    retrieval = RetrievalResult(blocks=blocks, trace=rt, context="【参考资料1】\n上下文内容")
    gen = GenerationResult(
        answer="答案 [1]。",
        citations=[Citation("答案 [1]。", 1, True, "d1", "d1:p0000", "上下文内容")],
        citation_stats={"n_sentences": 1, "n_sentences_with_marker": 1, "n_markers": 1, "n_markers_invalid": 0},
        model="deepseek-flash", prompt_version="v1",
        usage={"prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250, "cache_hit_tokens": 0},
        latency_ms=100.0,
    )
    return hyde, retrieval, gen


def test_build_trace_required_fields():
    hyde, retrieval, gen = _fixture_inputs()
    trace = build_trace(
        question="测试问题", config_id="cfg-test", hyde=hyde, retrieval=retrieval,
        generation=gen, prompt_version="v1", index_meta={"count": 1}, latency_total_ms=150.0,
    )
    for key in ["schema_version", "trace_id", "timestamp", "question", "config_id",
                "hyde", "vector", "bm25", "rrf", "rerank", "context", "generation", "totals", "versions"]:
        assert key in trace, key
    assert trace["schema_version"] == 1
    assert trace["generation"]["citations"][0]["marker"] == 1
    assert trace["totals"]["cost_estimated"] is True
    assert trace["totals"]["cost_total_cny"] > 0  # deepseek 生成有 token 成本
    assert trace["versions"]["gen_model"]
    json.dumps(trace, ensure_ascii=False)  # 必须可序列化


def test_trace_writer_appends(tmp_path):
    _, retrieval, gen = _fixture_inputs()
    trace = build_trace(question="q", config_id="c", retrieval=retrieval, generation=gen)
    path = tmp_path / "traces.jsonl"
    w = TraceWriter(path)
    w.append(trace)
    w.append(trace)
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2 and json.loads(lines[0])["question"] == "q"


def test_is_peak_time():
    assert is_peak_time(datetime(2026, 10, 5, 10, 0)) is True     # 周一上午高峰
    assert is_peak_time(datetime(2026, 10, 5, 12, 30)) is False   # 午休
    assert is_peak_time(datetime(2026, 10, 5, 15, 0)) is True     # 周二下午（周一例）
    assert is_peak_time(datetime(2026, 10, 10, 10, 0)) is False   # 周六全天空闲
