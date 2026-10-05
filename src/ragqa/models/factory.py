"""模型工厂：嵌入 / 重排 / 生成客户端（统一从这里取，配置走 .env + settings）。"""
from __future__ import annotations

from functools import lru_cache

from openai import AsyncOpenAI, OpenAI

from ragqa.config.settings import get_settings
from ragqa.utils.logging import logger
from ragqa.utils.paths import resolve_path


@lru_cache(maxsize=1)
def get_embed_model():
    """嵌入模型：api（DashScope）| local（本地 BGE，optional extra）。"""
    s = get_settings()
    if s.embed_mode == "local":
        from ragqa.models.local import LocalBGEEmbedding

        return LocalBGEEmbedding(resolve_path(s.embed_local_model))
    from ragqa.models.embedding import DashScopeEmbedding

    logger.info("[model] 嵌入: DashScope API / %s", s.embed_api_model)
    return DashScopeEmbedding()


@lru_cache(maxsize=1)
def get_reranker():
    """重排模型：api（DashScope）| local（CrossEncoder，optional extra）。"""
    s = get_settings()
    if s.rerank_mode == "local":
        from ragqa.models.local import LocalCrossEncoderReranker

        return LocalCrossEncoderReranker(resolve_path(s.rerank_local_model))
    from ragqa.models.rerank import DashScopeReranker

    logger.info("[model] 重排序: DashScope API / %s", s.rerank_api_model)
    return DashScopeReranker()


def get_gen_client() -> OpenAI:
    """生成模型客户端（OpenAI 兼容；默认 DeepSeek 官方）。"""
    s = get_settings()
    return OpenAI(api_key=s.gen_key, base_url=s.gen_url)


def get_async_gen_client() -> AsyncOpenAI:
    s = get_settings()
    return AsyncOpenAI(api_key=s.gen_key, base_url=s.gen_url)


def thinking_off_extra_body() -> dict:
    """"关闭思考"的请求参数：不同供应商字段不同（2026-10 实测）。

    - DeepSeek 官方：thinking={"type": "disabled"} 有效；enable_thinking 被忽略
    - DashScope（qwen3.x）：enable_thinking=False 有效
    - 未知供应商：两个都带上（实测两家均忽略未知字段而非报错）
    """
    s = get_settings()
    url = s.gen_url.lower()
    if "deepseek" in url:
        return {"thinking": {"type": "disabled"}}
    if "dashscope" in url or "aliyuncs" in url:
        return {"enable_thinking": False}
    return {"enable_thinking": False, "thinking": {"type": "disabled"}}
