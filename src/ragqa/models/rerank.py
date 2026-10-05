"""重排序客户端：DashScope rerank 服务化端点。

实测（2026-10）：qwen3.7-text-rerank 与旧 gte-rerank 端点/响应格式兼容：
  POST /api/v1/services/rerank/text-rerank/text-rerank
  -> {"output": {"results": [{"index": int, "relevance_score": float}, ...]}, "usage": {...}}

失败时不静默降级：抛 RerankError，由调用方决定回退策略并记入 trace。
"""
from __future__ import annotations

import time

import requests

from ragqa.config.settings import get_settings
from ragqa.utils.logging import logger

ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"


class RerankError(RuntimeError):
    pass


class DashScopeReranker:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        endpoint: str = ENDPOINT,
        max_retries: int = 3,
        timeout: int = 30,
        session: requests.Session | None = None,
    ) -> None:
        s = get_settings()
        self.model = model or s.rerank_api_model
        self._api_key = api_key or s.dashscope_api_key
        self.endpoint = endpoint
        self.max_retries = max_retries
        self.timeout = timeout
        self._session = session
        self.last_usage: dict | None = None
        self.last_latency_ms: float = 0.0

    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[dict]:
        """返回按相关性排序的 [{"index": int, "relevance_score": float}, ...]。"""
        if not documents:
            return []
        body = {
            "model": self.model,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": top_n or len(documents), "return_documents": False},
        }
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                post = self._session.post if self._session else requests.post
                resp = post(self.endpoint, json=body, headers=headers, timeout=self.timeout)
                self.last_latency_ms = (time.time() - t0) * 1000
                if resp.status_code == 200:
                    data = resp.json()
                    self.last_usage = data.get("usage")
                    return data.get("output", {}).get("results", [])
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_err = RerankError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    time.sleep(2 * (attempt + 1))
                    continue
                raise RerankError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            except RerankError:
                raise
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning("[rerank] 请求失败（第 %d/%d 次）：%s", attempt + 1, self.max_retries, e)
                time.sleep(2 * (attempt + 1))
        raise RerankError(f"重排调用失败（已重试 {self.max_retries} 次）：{last_err}")
