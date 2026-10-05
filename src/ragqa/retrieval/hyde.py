"""HyDE（Hypothetical Document Embeddings）：先生成假设性答案，再用它做检索。

失败不阻塞：任何异常都回退为原始 query，并在 HydeResult.error 中留痕（进 trace）。
"""
from __future__ import annotations

import time

from ragqa.config.loader import HydeConfig
from ragqa.models.factory import get_gen_client, thinking_off_extra_body
from ragqa.types import HydeResult
from ragqa.utils.llm import usage_dict
from ragqa.utils.logging import logger

HYDE_PROMPT = (
    "请根据以下问题，撰写一段详细、专业的中文段落来回答该问题。"
    "要求：内容详实、包含具体技术细节、用第三人称客观陈述、"
    "仿佛这是一篇教科书或技术文档中的原文。不要出现'根据问题'等元话语。\n\n"
    "问题：{query}"
)


def generate_hypothetical_doc(query: str, cfg: HydeConfig | None = None, client=None) -> HydeResult:
    cfg = cfg or HydeConfig()
    if not cfg.enabled:
        return HydeResult(enabled=False, used=False, query=query)

    t0 = time.time()
    try:
        client = client or get_gen_client()
        kwargs = {}
        if not cfg.enable_thinking:
            kwargs["extra_body"] = thinking_off_extra_body()
        resp = client.chat.completions.create(
            model=_gen_model(),
            messages=[{"role": "user", "content": HYDE_PROMPT.format(query=query)}],
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            **kwargs,
        )
        latency = (time.time() - t0) * 1000
        output = (resp.choices[0].message.content or "").strip()
        usage = usage_dict(resp)
        if output and len(output) > 10:
            return HydeResult(enabled=True, used=True, query=output, output=output, usage=usage, latency_ms=latency)
        return HydeResult(enabled=True, used=False, query=query, output=output or None,
                          error="HyDE 输出过短，回退原 query", usage=usage, latency_ms=latency)
    except Exception as e:  # noqa: BLE001 —— 降级设计：不阻塞主链路
        logger.warning("[HyDE] 生成失败，回退原 query: %s", e)
        return HydeResult(enabled=True, used=False, query=query, error=str(e), latency_ms=(time.time() - t0) * 1000)


def _gen_model() -> str:
    from ragqa.config.settings import get_settings

    return get_settings().gen_model
