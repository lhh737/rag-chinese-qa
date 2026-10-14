"""FastAPI 服务层测试（离线 stub 管道，不触网）。"""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from ragqa.api import create_app  # noqa: E402
from ragqa.types import (  # noqa: E402
    Citation,
    ContextBlock,
    QueryResult,
    RetrievalResult,
    RetrievalTrace,
    StreamChunk,
)


class _FakeStore:
    is_empty = False
    # meta 键对齐真实索引文件（index_meta.json: dim/embed_model/count/pipeline_version）
    meta = {"count": 32501, "dim": 1024, "embed_model": "test-embed", "pipeline_version": "test-v1"}


class _FakeCfg:
    def config_id(self) -> str:
        return "test-config"


class _FakeGen:
    prompt_version = "v1"


class FakePipeline:
    def __init__(self) -> None:
        self.store = _FakeStore()
        self.retrieval_cfg = _FakeCfg()
        self.generation_cfg = _FakeGen()

    def get_file_list(self) -> list[dict]:
        return [{"doc_id": "a", "source": "a.pdf"}]

    def get_document_content(self, doc_id: str) -> str | None:
        return "文档全文" if doc_id == "a" else None

    def query(self, question: str, prompt_version: str | None = None) -> QueryResult:
        return QueryResult(
            answer="根据参考资料，答案是 42[1]。",
            contexts=[ContextBlock(rank=1, parent_id="a:c1", doc_id="a", source="a.pdf", text="答案 42")],
            citations=[Citation(answer_sentence="答案是 42[1]。", marker=1, valid=True, source_doc="a")],
            trace={"config_id": "test-config"},
        )


class FakeStreamPipeline(FakePipeline):
    """流式门面（含 retrieval 阶段事件，与 core.astream 协议一致）。"""

    async def astream(self, question: str, prompt_version: str | None = None):
        ret = RetrievalResult(
            blocks=[ContextBlock(rank=1, parent_id="a:c1", doc_id="a", source="a.pdf",
                                 text="答案 42", rerank_score=0.91)],
            trace=RetrievalTrace(
                fusion={"strategy": "rrf", "k": 60, "n_vector": 24, "n_bm25": 24, "n_fused": 30},
                rerank={"scores": [0.91], "latency_ms": 12.5},
                latency_ms=40.0,
            ),
        )
        yield StreamChunk(stage="retrieval", retrieval=ret)
        yield StreamChunk(delta="根据")
        yield StreamChunk(delta="参考资料 [1]。")
        yield StreamChunk(done=True, result=self.query(question))


def test_health_and_documents():
    client = TestClient(create_app(FakePipeline()))
    h = client.get("/health").json()
    # n_docs 取自 get_file_list() 长度（本轮修复：原断言沿用了 meta 口径）
    assert h["status"] == "ok" and h["index"]["n_docs"] == 1
    assert h["index"]["n_chunks"] == 32501 and h["config_id"] == "test-config"
    d = client.get("/documents").json()
    assert d["files"][0]["doc_id"] == "a"


def test_query_endpoint():
    client = TestClient(create_app(FakePipeline()))
    r = client.post("/query", json={"question": "答案是什么？"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("根据参考资料")
    assert body["citations"][0]["marker"] == 1 and body["n_contexts"] == 1
    assert body["config_id"] == "test-config"


def test_query_validation():
    client = TestClient(create_app(FakePipeline()))
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_empty_store_503():
    class EmptyStore(_FakeStore):
        is_empty = True

    class EmptyPipeline(FakePipeline):
        def __init__(self):
            super().__init__()
            self.store = EmptyStore()

    client = TestClient(create_app(EmptyPipeline()))
    assert client.post("/query", json={"question": "x"}).status_code == 503


# ── Web UI 与流式端点（v2 界面）──────────────────────────

def _parse_sse(body: str) -> list[tuple[str, dict]]:
    import json

    events: list[tuple[str, dict]] = []
    for rec in body.split("\n\n"):
        rec = rec.strip()
        if not rec:
            continue
        ev, data = "", ""
        for line in rec.split("\n"):
            if line.startswith("event:"):
                ev = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data += line[len("data:"):].strip()
        if data:
            events.append((ev, json.loads(data)))
    return events


def test_query_stream_sse_sequence():
    """SSE 协议：retrieval → delta* → done（UI 溯源面板依赖此顺序）。"""
    client = TestClient(create_app(FakeStreamPipeline()))
    r = client.post("/query/stream", json={"question": "答案是什么？"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(r.text)
    assert [ev for ev, _ in events] == ["retrieval", "delta", "delta", "done"]

    ret = events[0][1]
    assert ret["funnel"]["n_vector"] == 24 and ret["funnel"]["n_context"] == 1
    assert ret["contexts"][0]["rerank_score"] == 0.91
    assert "".join(d["text"] for ev, d in events if ev == "delta") == "根据参考资料 [1]。"

    done = events[-1][1]
    assert done["answer"].startswith("根据参考资料")
    assert done["citations"][0]["valid"] is True


def test_document_content_endpoint():
    client = TestClient(create_app(FakePipeline()))
    assert client.get("/documents/a/content").json()["content"] == "文档全文"
    assert client.get("/documents/zzz/content").status_code == 404


def test_web_ui_served():
    client = TestClient(create_app(FakePipeline()))
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "引用溯源" in r.text
