"""测试公共设施：假嵌入 / 假重排 / 假生成客户端（离线、确定性，无需 API key）。"""
from __future__ import annotations

import math

import pytest


class FakeEmbedding:
    """确定性字符袋嵌入：同文本 → 同向量；相似文本 → 相近向量。"""

    dim = 64

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for ch in text:
            v[ord(ch) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]


class FakeReranker:
    """按字符重叠度打分的假重排（openai/DashScope 兼容返回格式）。"""

    def __init__(self) -> None:
        self.last_usage = {"prompt_tokens": 10, "total_tokens": 10}
        self.last_latency_ms = 1.0

    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[dict]:
        q = set(query)
        scored = [(i, len(q & set(d)) / max(len(q), 1)) for i, d in enumerate(documents)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [{"index": i, "relevance_score": s} for i, s in scored[: (top_n or len(documents))]]


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeUsage:
    prompt_tokens = 50
    completion_tokens = 10
    total_tokens = 60


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


class FakeGenClient:
    """同步假生成客户端（chat.completions.create 接口）。"""

    def __init__(self, answer: str = "这是测试答案 [1]。") -> None:
        self.answer = answer
        self.calls: list[dict] = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self.answer)


class _FakeStreamDelta:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeStreamChoice:
    def __init__(self, content: str) -> None:
        self.delta = _FakeStreamDelta(content)


class _FakeStreamChunk:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeStreamChoice(content)]


class _FakeAsyncCompletions:
    def __init__(self, parts: list[str]) -> None:
        self.parts = parts

    async def create(self, **kwargs):
        parts = self.parts

        async def _gen():
            for p in parts:
                yield _FakeStreamChunk(p)

        return _gen()


class FakeAsyncGenClient:
    """异步假生成客户端（流式）。"""

    def __init__(self, parts: list[str] | None = None) -> None:
        self.parts = parts or ["这是", "测试", "答案 [1]。"]
        self.chat = self
        self.completions = _FakeAsyncCompletions(self.parts)


@pytest.fixture
def fake_embed() -> FakeEmbedding:
    return FakeEmbedding()


@pytest.fixture
def fake_reranker() -> FakeReranker:
    return FakeReranker()
