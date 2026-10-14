"""FastAPI 服务层测试（离线 stub 管道，不触网）。"""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from ragqa.api import create_app  # noqa: E402
from ragqa.types import Citation, ContextBlock, QueryResult  # noqa: E402


class _FakeStore:
    is_empty = False
    meta = {"n_docs": 106, "n_chunks": 32501, "dim": 1024, "embed_model": "test-embed"}


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

    def query(self, question: str, prompt_version: str | None = None) -> QueryResult:
        return QueryResult(
            answer="根据参考资料，答案是 42[1]。",
            contexts=[ContextBlock(rank=1, parent_id="a:c1", doc_id="a", source="a.pdf", text="答案 42")],
            citations=[Citation(answer_sentence="答案是 42[1]。", marker=1, valid=True, source_doc="a")],
            trace={"config_id": "test-config"},
        )


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
