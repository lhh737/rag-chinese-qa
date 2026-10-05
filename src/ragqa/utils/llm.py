"""LLM 响应工具：usage 归一化（跨供应商字段差异收敛到这里）。"""
from __future__ import annotations


def usage_dict(resp) -> dict[str, int]:
    """把 OpenAI 兼容响应的 usage 归一化为 dict（含缓存命中与推理 token）。"""
    u = getattr(resp, "usage", None)
    if not u:
        return {}
    out: dict[str, int] = {
        "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
        "total_tokens": getattr(u, "total_tokens", 0) or 0,
    }
    details = getattr(u, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None) if details else None
    if reasoning:
        out["reasoning_tokens"] = int(reasoning)
    cache_hit = getattr(u, "prompt_cache_hit_tokens", None)
    if cache_hit is not None:
        out["cache_hit_tokens"] = int(cache_hit)
    return out
