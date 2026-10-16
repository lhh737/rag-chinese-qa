"""单轮工具选择变体（P9-D）：让模型自己决定检索什么，其余全部复用现有管线。

与固定管线的**唯一差异在检索决策**：固定管线无条件走一次检索；本变体先做 **一次**
function-calling 轮，由模型决定发出几个 `search_documents` 调用（0..N —— N 次即多查询检索），
执行后按 RRF 跨子查询合并，再走同一个 `generation.generator.generate`。

**定位：一次可判定的实验，正负结果都留档**（见 README「统计化迭代」）。假设来源是 P5 归因
留档的最大缺口——多跳题「第二来源从未召回」属候选池层问题，README 已注明「需多查询检索层工作」。

设计约束：
- **不改 `RAGPipeline`**：本类鸭子类型实现 `query()`，评测器只依赖这一个契约；
- **不改 `generate`**：生成仍走同一入口，故 judge / 引用解析口径完全一致；
- **不做 ReAct 多轮循环**：只有一次工具选择轮，成本与延迟可界；
- **单查询不重排**：只发出一个检索调用时直接沿用其结果，避免合并逻辑引入无谓扰动。

trace 复用 `trace.build_trace`，并把工具选择轮的令牌 / 成本 / 延迟并入 `totals`——
否则 agentic 的成本对比会被系统性低估。
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Any

from ragqa.config.loader import RetrievalConfig
from ragqa.config.settings import get_settings
from ragqa.core import RAGPipeline
from ragqa.generation.generator import generate
from ragqa.models.factory import thinking_off_extra_body
from ragqa.retrieval.hybrid import dynamic_parent_k, format_context
from ragqa.trace import build_trace, estimate_gen_cost
from ragqa.tools import FUNCTION_SPECS
from ragqa.types import ContextBlock, QueryResult, RetrievalResult, RetrievalTrace
from ragqa.utils.llm import usage_dict
from ragqa.utils.logging import logger

# 只提供检索工具：单轮 QA 场景下「取全文」几乎不是正确选择，且会撑爆上下文窗口。
# 多查询通过**一次发出多个 search_documents 调用**表达（比另设 multi_query 工具更贴近
# function calling 的真实用法，也避免工具集膨胀）。
_TOOL_SELECT_SYSTEM = """你是检索助手，唯一的工具是 search_documents，用于从中文论文/文档库检索回答所需的资料。

规则：
1. 需要事实依据的问题必须先检索，不要凭记忆回答。
2. 问题涉及多个主体、需要横向对比、或需要多跳推理时，请**在一次回复中发出多个** search_documents 调用，
   每个调用针对一个具体方面、使用不同措辞的查询；单一直接问题只发一个调用。
3. 只做检索决策，不要回答用户的问题。"""

TOOL_SELECT_PROMPT_VERSION = "toolsel-v1"


def _sum_usage(usages: list[dict | None]) -> dict | None:
    """合并多次调用的 usage（逐字段求和），供成本按同一口径计算。"""
    out: dict[str, float] = {}
    for u in usages:
        if not u:
            continue
        for k, v in u.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[k] = out.get(k, 0) + v
    return out or None


def _parse_tool_calls(message: Any) -> list[tuple[str, dict]]:
    """从模型回复中取出工具调用；参数非法 JSON 的调用直接丢弃（不阻塞主链路）。"""
    calls: list[tuple[str, dict]] = []
    for c in getattr(message, "tool_calls", None) or []:
        fn = getattr(c, "function", None)
        name = getattr(fn, "name", None)
        raw = getattr(fn, "arguments", "") or "{}"
        if not name:
            continue
        try:
            args = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning("[agentic] 工具参数非合法 JSON，丢弃该调用: %r", raw)
            continue
        calls.append((name, args if isinstance(args, dict) else {}))
    return calls


class ToolCallingPipeline:
    """鸭子类型门面：实现 EvaluationRunner 需要的 `query(question) -> QueryResult`。"""

    def __init__(
        self,
        base: RAGPipeline,
        *,
        gen_client: Any | None = None,
        max_tool_calls: int = 4,
    ) -> None:
        self.base = base
        self.retrieval_cfg = base.retrieval_cfg
        self.generation_cfg = base.generation_cfg
        self.store = base.store
        self.retriever = base.retriever
        self.max_tool_calls = max_tool_calls
        self._gen_client = gen_client

    # ── 工具选择轮 ────────────────────────────────────────

    def _client(self) -> Any:
        """工具选择轮与生成轮共用同一客户端：优先显式注入，其次沿用基管线的注入。"""
        if self._gen_client is None:
            self._gen_client = self.base._gen_client
        if self._gen_client is None:
            from ragqa.models.factory import get_gen_client

            self._gen_client = get_gen_client()
        return self._gen_client

    def _select_tools(self, question: str) -> tuple[list[tuple[str, dict]], dict | None, float]:
        kwargs: dict[str, Any] = {
            "model": get_settings().gen_model,
            "messages": [
                {"role": "system", "content": _TOOL_SELECT_SYSTEM},
                {"role": "user", "content": question},
            ],
            "temperature": 0.0,
            "max_tokens": 512,
            "tools": [s for s in FUNCTION_SPECS if s["function"]["name"] == "search_documents"],
            "tool_choice": "auto",
            "extra_body": thinking_off_extra_body(),
        }
        t0 = time.time()
        resp = self._client().chat.completions.create(**kwargs)
        latency_ms = (time.time() - t0) * 1000
        calls = _parse_tool_calls(resp.choices[0].message)
        return calls, usage_dict(resp), latency_ms

    # ── 多子查询合并（RRF 跨查询）─────────────────────────

    def _merge(self, question: str, subs: list[RetrievalResult], cfg: RetrievalConfig) -> RetrievalResult:
        """把多个子查询结果合并为一个 `RetrievalResult`。

        子块层：按 `score(u) = Σ_q 1/(k + rank_q(u))` 跨查询 RRF——与单查询 RRF 同式同 k。
        父块层：并集去重（同 parent 取较高重排分）→ 按重排分排序 → 同一条 `dynamic_parent_k` 取量。
        """
        k = cfg.rrf_k
        scores: dict[str, float] = {}
        detail: dict[str, dict] = {}
        for qi, rr in enumerate(subs, start=1):
            for rank, f in enumerate((rr.trace.fusion or {}).get("fused") or [], start=1):
                uid = str(f.get("chunk_uid", ""))
                if not uid:
                    continue
                scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank)
                detail.setdefault(uid, {})[f"q{qi}_rank"] = rank
        ordered = sorted(scores, key=lambda u: scores[u], reverse=True)

        by_parent: dict[str, ContextBlock] = {}
        for rr in subs:
            for b in rr.blocks:
                prev = by_parent.get(b.parent_id)
                if prev is None or (b.rerank_score or 0.0) > (prev.rerank_score or 0.0):
                    by_parent[b.parent_id] = b
        ranked = sorted(by_parent.values(), key=lambda b: (b.rerank_score or 0.0), reverse=True)
        take = dynamic_parent_k(question, [b.rerank_score or 0.0 for b in ranked], cfg.base_context_k, cfg.max_context_k)
        blocks = [replace(b, rank=i) for i, b in enumerate(ranked[:take], start=1)]

        context = format_context(blocks)
        trace = RetrievalTrace(
            vector_hits=[h for rr in subs for h in rr.trace.vector_hits],
            bm25_hits=[h for rr in subs for h in rr.trace.bm25_hits],
            fusion={
                "strategy": "rrf_multiquery",
                "k": k,
                "n_subqueries": len(subs),
                "n_vector": sum(len(rr.trace.vector_hits) for rr in subs),
                "n_bm25": sum(len(rr.trace.bm25_hits) for rr in subs),
                "n_fused": len(ordered),
                "fused": [{"chunk_uid": u, "rrf_score": round(scores[u], 6), **detail[u]} for u in ordered],
            },
            rerank={
                "scores": [round(b.rerank_score or 0.0, 6) for b in blocks],
                "top": [b.parent_id for b in blocks],
                "latency_ms": round(sum(rr.trace.latency_ms for rr in subs), 1),
                "usage": _sum_usage([rr.trace.rerank.get("usage") if rr.trace.rerank else None for rr in subs]),
                "error": None,
                "merged_from": len(subs),
            },
            context_parent_ids=[b.parent_id for b in blocks],
            context_chars=len(context),
            latency_ms=round(sum(rr.trace.latency_ms for rr in subs), 1),
        )
        return RetrievalResult(blocks=blocks, trace=trace, context=context)

    # ── 查询（EvaluationRunner 契约）──────────────────────

    def query(
        self,
        question: str,
        *,
        config: RetrievalConfig | None = None,
        prompt_version: str | None = None,
    ) -> QueryResult:
        cfg = config or self.base.retrieval_cfg
        t0 = time.perf_counter()

        if self.base.store.is_empty:
            return self.base.query(question, config=cfg, prompt_version=prompt_version)

        sel_error: str | None = None
        try:
            calls, sel_usage, sel_latency = self._select_tools(question)
        except Exception as e:  # noqa: BLE001 — 工具选择轮失败不阻塞主链路，回退固定管线
            logger.warning("[agentic] 工具选择失败，回退固定管线: %s", e)
            sel_error, calls, sel_usage, sel_latency = str(e), [], None, 0.0

        sub_queries: list[str] = []
        subs: list[RetrievalResult] = []
        for name, args in calls[: self.max_tool_calls]:
            if name != "search_documents":
                continue
            q = str(args.get("query") or "").strip()
            if not q:
                continue
            sub_queries.append(q)
            try:
                subs.append(self.base.retriever.retrieve(q))
            except Exception as e:  # noqa: BLE001 — 单个子查询失败不影响其余
                logger.warning("[agentic] 子查询检索失败，跳过: %s", e)

        if not subs:
            retrieval = None
            answer = "未找到相关文档内容。"
            generation = None
        else:
            retrieval = subs[0] if len(subs) == 1 else self._merge(question, subs, cfg)
            if retrieval.blocks:
                generation = generate(
                    question,
                    retrieval.context,
                    retrieval.blocks,
                    prompt_version=prompt_version,
                    cfg=self.generation_cfg,
                    client=self._client(),
                )
                answer = generation.answer
            else:
                answer = "未找到相关文档内容。"
                generation = None

        config_id = f"{cfg.config_id()}|{TOOL_SELECT_PROMPT_VERSION}"
        trace = build_trace(
            question=question,
            config_id=config_id,
            hyde=None,  # 本变体不使用 HyDE
            retrieval=retrieval,
            generation=generation,
            prompt_version=prompt_version or self.generation_cfg.prompt_version,
            index_meta=self.base.store.meta,
            latency_total_ms=(time.perf_counter() - t0) * 1000,
        )
        # 工具选择轮的成本/延迟/令牌并入 totals（否则 agentic 成本被系统性低估）
        if sel_usage:
            extra = estimate_gen_cost(get_settings().gen_model, sel_usage)
            trace["totals"]["cost_by_stage_cny"]["tool_select"] = round(extra, 6)
            trace["totals"]["cost_total_cny"] = round(trace["totals"]["cost_total_cny"] + extra, 6)
            trace["totals"]["tokens"]["tool_select"] = int(sel_usage.get("total_tokens", 0) or 0)
        trace["totals"]["latency_ms"]["tool_select"] = round(sel_latency, 1)
        trace["agentic"] = {
            "mode": "single_turn_tool_selection",
            "prompt_version": TOOL_SELECT_PROMPT_VERSION,
            "n_tool_calls": len(calls),
            "tool_calls": [{"name": n, "arguments": a} for n, a in calls],
            "sub_queries": sub_queries,
            "tool_select_usage": sel_usage,
            "tool_select_latency_ms": round(sel_latency, 1),
            "tool_select_error": sel_error,
        }

        return QueryResult(
            answer=answer,
            contexts=retrieval.blocks if retrieval else [],
            citations=generation.citations if generation else [],
            trace=trace,
        )
