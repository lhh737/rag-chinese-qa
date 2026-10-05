"""生成层：prompt 注册表 + 行内引用 + 同步生成 / 异步流式。

- 模型：OpenAI 兼容端点（默认 DeepSeek 官方，见 .env GEN_*）
- 输出：事实性陈述带 [n] 行内引用（解析见 citations.py）
- 思考开关：混合思考模型默认关闭（models.factory.thinking_off_extra_body）
"""
from __future__ import annotations

import time
from typing import Any, AsyncIterator

from ragqa.config.loader import GenerationConfig, load_generation_config
from ragqa.config.settings import get_settings
from ragqa.generation.citations import parse_citations
from ragqa.generation.prompts import load_answer_prompt
from ragqa.models.factory import (
    get_async_gen_client,
    get_gen_client,
    thinking_off_extra_body,
)
from ragqa.types import ContextBlock, GenerationResult
from ragqa.utils.llm import usage_dict


def build_messages(question: str, context: str, prompt_version: str) -> list[dict]:
    system = load_answer_prompt(prompt_version).format(context=context)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def request_kwargs(cfg: GenerationConfig, messages: list[dict]) -> dict:
    kwargs: dict[str, Any] = {
        "model": get_settings().gen_model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    if not cfg.enable_thinking:
        kwargs["extra_body"] = thinking_off_extra_body()
    return kwargs


def generate(
    question: str,
    context: str,
    contexts: list[ContextBlock],
    *,
    prompt_version: str | None = None,
    cfg: GenerationConfig | None = None,
    client: Any | None = None,
) -> GenerationResult:
    """同步生成：返回答案 + 逐句引用解析结果 + usage/latency。"""
    cfg = cfg or load_generation_config()
    prompt_version = prompt_version or cfg.prompt_version
    client = client or get_gen_client()

    t0 = time.time()
    resp = client.chat.completions.create(**request_kwargs(cfg, build_messages(question, context, prompt_version)))
    latency_ms = (time.time() - t0) * 1000

    answer = (resp.choices[0].message.content or "").strip()
    citations, stats = parse_citations(answer, contexts)
    return GenerationResult(
        answer=answer,
        citations=citations,
        citation_stats=stats,
        model=get_settings().gen_model,
        prompt_version=prompt_version,
        usage=usage_dict(resp),
        latency_ms=latency_ms,
    )


class GenerationSession:
    """异步流式生成会话。

    用法：
        session = GenerationSession(question, context, contexts)
        async for delta in session:
            ...  # 展示用增量
        result = session.result  # 完整 GenerationResult（含引用解析）

    注意：流式下部分供应商不返回 usage（usage 记空，成本口径以非流式为准）。
    """

    def __init__(
        self,
        question: str,
        context: str,
        contexts: list[ContextBlock],
        *,
        prompt_version: str | None = None,
        cfg: GenerationConfig | None = None,
        client: Any | None = None,
    ) -> None:
        self.cfg = cfg or load_generation_config()
        self.prompt_version = prompt_version or self.cfg.prompt_version
        self.question = question
        self.context = context
        self.contexts = contexts
        self._client = client
        self.result: GenerationResult | None = None
        self._started = 0.0

    def _stream_kwargs(self) -> dict:
        messages = build_messages(self.question, self.context, self.prompt_version)
        kwargs = request_kwargs(self.cfg, messages)
        kwargs["stream"] = True
        return kwargs

    async def _iter_deltas(self) -> AsyncIterator[str]:
        client = self._client or get_async_gen_client()
        stream = await client.chat.completions.create(**self._stream_kwargs())
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[str]:
        self._started = time.time()
        parts: list[str] = []
        async for delta in self._iter_deltas():
            parts.append(delta)
            yield delta
        answer = "".join(parts).strip()
        citations, stats = parse_citations(answer, self.contexts)
        self.result = GenerationResult(
            answer=answer,
            citations=citations,
            citation_stats=stats,
            model=get_settings().gen_model,
            prompt_version=self.prompt_version,
            usage={},  # 流式 usage 视供应商而定，暂记空
            latency_ms=(time.time() - self._started) * 1000,
        )
