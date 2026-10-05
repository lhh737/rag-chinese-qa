"""DashScope 文本嵌入客户端（OpenAI 兼容模式）。

- 鸭子类型接口 `embed_documents / embed_query`（不依赖 langchain 基类）
- 分批调用（默认 batch=10）+ 退避重试（429 / 5xx / 超时）
- 累计 token 用量（`total_tokens`），供入库成本核算与 trace 记账
"""
from __future__ import annotations

import time

from openai import OpenAI

from ragqa.config.settings import get_settings
from ragqa.utils.logging import logger

_RETRYABLE_MARKERS = ("429", "rate", "timeout", "timed out", "connection", "overloaded",
                      "500", "502", "503", "504", "internal error")


def _is_retryable(err: Exception) -> bool:
    status = getattr(err, "status_code", None)
    if isinstance(status, int) and (status == 429 or status >= 500):
        return True
    msg = f"{type(err).__name__}: {err}".lower()
    return any(m in msg for m in _RETRYABLE_MARKERS)


class DashScopeEmbedding:
    def __init__(
        self,
        model: str | None = None,
        dimensions: int | None = None,
        batch_size: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        client: OpenAI | None = None,
        max_retries: int = 4,
    ) -> None:
        s = get_settings()
        self.model = model or s.embed_api_model
        self.dimensions = dimensions or s.embed_dimensions
        self.batch_size = batch_size or s.embed_batch_size
        self._api_key = api_key or s.dashscope_api_key
        self._base_url = base_url or s.llm_base_url
        self._client = client
        self.max_retries = max_retries
        self.total_tokens = 0

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    # ── 内部 ──────────────────────────────────────────────

    def _embed_batch(self, batch: list[str], text_type: str) -> list[list[float]]:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimensions,
                    extra_body={"text_type": text_type},
                )
                self.total_tokens += getattr(resp.usage, "total_tokens", 0) or 0
                ordered = {i: item.embedding for i, item in enumerate(resp.data)}
                return [ordered.get(i, [0.0] * self.dimensions) for i in range(len(batch))]
            except Exception as e:  # noqa: BLE001 —— 需按可重试性分流
                last_err = e
                if attempt == self.max_retries - 1 or not _is_retryable(e):
                    raise
                wait = 2 ** attempt
                logger.warning("[embed] 第 %d 次失败（%s），%.0fs 后重试", attempt + 1, type(e).__name__, wait)
                time.sleep(wait)
        raise last_err  # pragma: no cover

    def _embed(self, texts: list[str], text_type: str) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            # 空串不调用 API，返回零向量以保持对齐（正常分块不会产生空串）
            if all(not t.strip() for t in batch):
                vectors.extend([[0.0] * self.dimensions for _ in batch])
                continue
            batch_clean = [t if t.strip() else " " for t in batch]
            vectors.extend(self._embed_batch(batch_clean, text_type))
        return vectors

    # ── 对外接口（鸭子类型）──────────────────────────────

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, text_type="document")

    def embed_query(self, text: str) -> list[float]:
        result = self._embed([text], text_type="query")
        return result[0] if result else [0.0] * self.dimensions
