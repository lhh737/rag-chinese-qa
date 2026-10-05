"""嵌入客户端测试：分批、参数透传、供应商差异（离线假客户端）。"""
from __future__ import annotations

import pytest

from ragqa.models.embedding import DashScopeEmbedding, OpenAIEmbedding


class _Item:
    def __init__(self, vec: list[float]) -> None:
        self.embedding = vec


class _Usage:
    total_tokens = 7


class _Resp:
    def __init__(self, n: int, dim: int) -> None:
        self.data = [_Item([0.1] * dim) for _ in range(n)]
        self.usage = _Usage()


class FakeEmbeddingClient:
    """记录每次调用的 kwargs，返回固定维度向量。"""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.calls: list[dict] = []
        self.embeddings = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Resp(len(kwargs["input"]), self.dim)


def test_dashscope_text_type_and_dimensions():
    client = FakeEmbeddingClient(dim=1024)
    e = DashScopeEmbedding(client=client, dimensions=1024, batch_size=10)
    e.embed_query("问题")
    e.embed_documents(["文档一", "文档二"])
    assert client.calls[0]["extra_body"] == {"text_type": "query"}
    assert client.calls[0]["dimensions"] == 1024
    assert client.calls[1]["extra_body"] == {"text_type": "document"}


def test_openai_no_dimensions_when_zero():
    client = FakeEmbeddingClient(dim=2560)
    e = OpenAIEmbedding(client=client, dimensions=0)
    vecs = e.embed_documents(["文本"])
    assert len(vecs[0]) == 2560
    assert "dimensions" not in client.calls[0]
    assert "extra_body" not in client.calls[0]
    # 部分端点只接受 float（openai SDK 默认 base64，会 400）
    assert client.calls[0]["encoding_format"] == "float"


def test_openai_dimensions_and_extra_body_passed():
    client = FakeEmbeddingClient()
    e = OpenAIEmbedding(client=client, dimensions=1024, extra_body={"foo": "bar"})
    e.embed_query("问题")
    assert client.calls[0]["dimensions"] == 1024
    assert client.calls[0]["extra_body"] == {"foo": "bar"}
    assert client.calls[0]["model"]  # 模型名必传


def test_batching():
    client = FakeEmbeddingClient()
    e = OpenAIEmbedding(client=client, dimensions=0, batch_size=10)
    vecs = e.embed_documents([f"文本{i}" for i in range(25)])
    assert len(vecs) == 25
    assert [len(c["input"]) for c in client.calls] == [10, 10, 5]


def test_blank_replaced_not_skipped():
    client = FakeEmbeddingClient()
    e = OpenAIEmbedding(client=client, dimensions=0, batch_size=10)
    vecs = e.embed_documents(["", "  ", "正常"])
    assert len(vecs) == 3
    assert client.calls[0]["input"] == [" ", " ", "正常"]


def test_empty_input_returns_empty():
    client = FakeEmbeddingClient()
    e = OpenAIEmbedding(client=client, dimensions=0)
    assert e.embed_documents([]) == []
    assert client.calls == []


def test_retryable_errors():
    from ragqa.models.embedding import _is_retryable

    class E429(Exception):
        status_code = 429

    class E400(Exception):
        status_code = 400

    assert _is_retryable(E429("rate limit"))
    assert not _is_retryable(E400("bad request"))
