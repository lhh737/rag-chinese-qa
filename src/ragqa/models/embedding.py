"""嵌入客户端（供应商可切换，接口统一为鸭子类型 embed_documents / embed_query）。

- `DashScopeEmbedding`：阿里云百炼 OpenAI 兼容端点（区分 text_type=document/query）
- `OpenAIEmbedding`：通用 OpenAI 兼容端点（腾讯 MaaS TokenHub · Kinfra 等）

共同能力：分批调用（默认 batch=10）+ 退避重试（429 / 5xx / 超时）+ 累计 token 用量。
供应商通过 .env `EMBED_PROVIDER` 切换（注释/取消注释即可换模型）。
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable

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


def call_with_retry(fn: Callable[[], Any], max_retries: int, label: str):
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 —— 按可重试性分流
            last_err = e
            if attempt == max_retries - 1 or not _is_retryable(e):
                raise
            wait = 2 ** attempt
            logger.warning("[embed:%s] 第 %d 次失败（%s），%.0fs 后重试", label, attempt + 1, type(e).__name__, wait)
            time.sleep(wait)
    raise last_err  # pragma: no cover


class _BatchedEmbedding:
    """分批 + 空白串替换的公共骨架。"""

    batch_size: int = 10

    def _embed(self, texts: list[str], **batch_kwargs) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t if t.strip() else " " for t in texts[start : start + self.batch_size]]
            vectors.extend(self._embed_batch(batch, **batch_kwargs))
        return vectors

    def _embed_batch(self, batch: list[str], **kwargs) -> list[list[float]]:
        raise NotImplementedError

    # 鸭子类型接口
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


class DashScopeEmbedding(_BatchedEmbedding):
    """阿里云百炼（text_type 区分 document / query）。"""

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
        self.dimensions = dimensions or s.embed_dimensions or 1024
        self.batch_size = batch_size or s.embed_batch_size
        self._api_key = api_key or s.embed_api_key or s.dashscope_api_key
        self._base_url = base_url or s.embed_base_url or s.llm_base_url
        self._client = client
        self.max_retries = max_retries
        self.total_tokens = 0

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def _embed_batch(self, batch: list[str], text_type: str = "document") -> list[list[float]]:
        def _do():
            return self.client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dimensions,
                extra_body={"text_type": text_type},
            )

        resp = call_with_retry(_do, self.max_retries, "dashscope")
        self.total_tokens += getattr(resp.usage, "total_tokens", 0) or 0
        ordered = {i: item.embedding for i, item in enumerate(resp.data)}
        return [ordered.get(i, [0.0] * self.dimensions) for i in range(len(batch))]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, text_type="document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], text_type="query")[0]


class OpenAIEmbedding(_BatchedEmbedding):
    """通用 OpenAI 兼容嵌入端点（腾讯 MaaS TokenHub · Kinfra / 其他）。

    - `dimensions=0` 时不发送 dimensions 参数（用模型原生维度，如 Kinfra-4b=2560）
    - `extra_body`：供应商特有参数（JSON dict）
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        dimensions: int | None = None,
        extra_body: dict | None = None,
        batch_size: int | None = None,
        client: OpenAI | None = None,
        max_retries: int = 4,
    ) -> None:
        s = get_settings()
        self.model = model or s.embed_api_model
        self.dimensions = s.embed_dimensions if dimensions is None else dimensions
        self.batch_size = batch_size or s.embed_batch_size
        self._base_url = base_url or s.embed_base_url or s.llm_base_url
        self._api_key = api_key or s.embed_api_key or s.dashscope_api_key
        if extra_body is None and s.embed_extra_body.strip():
            extra_body = json.loads(s.embed_extra_body)  # 配置错误要响亮失败
        self._extra_body = extra_body
        self._client = client
        self.max_retries = max_retries
        self.total_tokens = 0

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def _embed_batch(self, batch: list[str], **kwargs) -> list[list[float]]:
        # openai SDK 默认以 base64 传 encoding_format，部分端点（如 TokenHub）只接受 float
        params: dict[str, Any] = {"model": self.model, "input": batch, "encoding_format": "float"}
        if self.dimensions:
            params["dimensions"] = self.dimensions
        if self._extra_body:
            params["extra_body"] = self._extra_body

        def _do():
            return self.client.embeddings.create(**params)

        resp = call_with_retry(_do, self.max_retries, "openai")
        self.total_tokens += getattr(resp.usage, "total_tokens", 0) or 0
        ordered = {i: item.embedding for i, item in enumerate(resp.data)}
        fallback_dim = len(next(iter(ordered.values()))) if ordered else 0
        return [ordered.get(i, [0.0] * fallback_dim) for i in range(len(batch))]
