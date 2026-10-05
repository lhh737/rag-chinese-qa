"""
DashScope text-embedding-v4 嵌入封装。

用 openai SDK 调用阿里云百炼兼容模式，适配 LangChain Embeddings 接口。
"""
from __future__ import annotations

from typing import Any

import numpy as np
from langchain_core.embeddings import Embeddings
from openai import OpenAI

from config.settings import DASHSCOPE_API_KEY, LLM_BASE_URL


class DashScopeTextEmbedding(Embeddings):
    """阿里云百炼文本嵌入（DashScope API）。"""

    def __init__(self, model: str = "text-embedding-v3", dimensions: int = 1024):
        self.model = model
        self.dimensions = dimensions
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=LLM_BASE_URL)
        return self._client

    MAX_BATCH = 10  # DashScope API 限制

    def _embed(self, texts: list[str], text_type: str = "document") -> list[list[float]]:
        valid = [t for t in texts if t and t.strip()]
        if not valid:
            return []
        # 分片调用，每片不超过 MAX_BATCH 条
        all_embeddings: list[list[float]] = []
        for start in range(0, len(valid), self.MAX_BATCH):
            batch = valid[start : start + self.MAX_BATCH]
            resp = self.client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dimensions,
                extra_body={"text_type": text_type},
            )
            ordered = {i: e.embedding for i, e in enumerate(resp.data)}
            all_embeddings.extend(
                ordered.get(i, [0.0] * self.dimensions) for i in range(len(batch))
            )
        return all_embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, text_type="document")

    def embed_query(self, text: str) -> list[float]:
        result = self._embed([text], text_type="query")
        return result[0] if result else [0.0] * self.dimensions
