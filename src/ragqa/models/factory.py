"""模型工厂：API (DashScope) 优先，本地 BGE 可通过 .env 切换。"""
from __future__ import annotations

import os
from functools import lru_cache

from config.settings import (
    EMBED_MODE, EMBED_API_MODEL, EMBED_LOCAL_MODEL,
    RERANK_MODE, RERANK_API_MODEL, RERANK_LOCAL_MODEL,
    DASHSCOPE_API_KEY,
)
from utils.logger_handler import logger
from utils.path_tool import get_project_root


# ── 嵌入模型 ──────────────────────────────────────────────

def _create_api_embedding():
    from model.dashscope_embedding import DashScopeTextEmbedding
    logger.info("[model] 嵌入: DashScope API / %s", EMBED_API_MODEL)
    return DashScopeTextEmbedding(model=EMBED_API_MODEL)


def _create_local_embedding():
    from langchain_huggingface import HuggingFaceEmbeddings
    model_path = EMBED_LOCAL_MODEL
    if not os.path.isabs(model_path):
        model_path = os.path.join(get_project_root(), model_path)
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(f"本地嵌入模型不完整: {model_path}")
    logger.info("[model] 嵌入: 本地 BGE / %s", model_path)
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cpu", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 4},
    )


@lru_cache(maxsize=1)
def get_embed_model():
    if EMBED_MODE == "local":
        return _create_local_embedding()
    return _create_api_embedding()


# ── 重排序模型 ────────────────────────────────────────────

class DashScopeReranker:
    """DashScope gte-rerank API 封装。"""

    def __init__(self, model: str = "gte-rerank-v2"):
        self.model = model

    def rerank(self, query: str, documents: list[str], top_n: int = 10) -> list[dict]:
        import requests
        import time

        url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
        headers = {
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": top_n, "return_documents": False},
        }

        for attempt in range(3):
            try:
                resp = requests.post(url, json=body, headers=headers, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("output", {}).get("results", [])
                if resp.status_code == 429:
                    time.sleep(2 * (attempt + 1))
                    continue
                logger.warning("[rerank] API 错误 %d: %s", resp.status_code, resp.text[:200])
                break
            except Exception as e:
                logger.warning("[rerank] 请求失败 (重试 %d/3): %s", attempt + 1, e)
                time.sleep(2)
        return []


def _create_api_reranker():
    logger.info("[model] 重排序: DashScope API / %s", RERANK_API_MODEL)
    return DashScopeReranker(model=RERANK_API_MODEL)


def _create_local_reranker():
    from sentence_transformers import CrossEncoder
    model_path = RERANK_LOCAL_MODEL
    if not os.path.isabs(model_path):
        model_path = os.path.join(get_project_root(), model_path)
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(f"本地重排序模型不完整: {model_path}")
    logger.info("[model] 重排序: 本地 BGE / %s", model_path)
    return CrossEncoder(model_path, device="cpu")


@lru_cache(maxsize=1)
def get_reranker():
    if RERANK_MODE == "local":
        return _create_local_reranker()
    return _create_api_reranker()
