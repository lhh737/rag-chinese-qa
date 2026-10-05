"""Trace Schema v1：评测 / 错误分析 / 归因的唯一事实来源（设计 §5.1）。

一条 trace 记录一次查询的全部可回溯信息：
question / config_id / hyde / vector / bm25 / rrf / rerank / context / generation / totals / versions

- TraceWriter：JSONL append 落盘（P1 评测 runner 复用）
- 成本估算：config/pricing.yml 单价 + 峰谷时段 —— 口径为"估算"，报告须注明
"""
from __future__ import annotations

import json
import subprocess
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ragqa.config.loader import PricingConfig, load_pricing_config
from ragqa.config.settings import get_settings
from ragqa.types import GenerationResult, HydeResult, RetrievalResult
from ragqa.utils.paths import REPO_ROOT

TRACE_SCHEMA_VERSION = 1

_commit_cache: str | None = None


def git_commit() -> str:
    """当前代码版本（短 sha）；不在 git 仓库时返回 unknown。"""
    global _commit_cache
    if _commit_cache is None:
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=5,
            )
            _commit_cache = proc.stdout.strip() or "unknown"
        except Exception:
            _commit_cache = "unknown"
    return _commit_cache


def new_trace_id() -> str:
    return f"t{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def is_peak_time(now: datetime | None = None) -> bool:
    """DeepSeek 峰谷计价：高峰=北京时间周一至周五 9:00-12:00 / 14:00-18:00。"""
    now = now or datetime.now()
    if now.weekday() >= 5:
        return False
    return 9 <= now.hour < 12 or 14 <= now.hour < 18


# ── 成本估算（元）──────────────────────────────────────────

def _gen_cost(model: str, usage: dict, pricing: PricingConfig, peak: bool) -> float:
    p = pricing.generation.get(model) or pricing.generation.get("_default") or {}
    in_rate = p.get("input_peak" if peak else "input_idle", 0.0)
    out_rate = p.get("output_peak" if peak else "output_idle", 0.0)
    hit_rate = p.get("cache_hit_peak" if peak else "cache_hit_idle", in_rate)
    prompt = int(usage.get("prompt_tokens", 0))
    completion = int(usage.get("completion_tokens", 0))
    cache_hit = int(usage.get("cache_hit_tokens", 0))
    return ((prompt - cache_hit) * in_rate + cache_hit * hit_rate + completion * out_rate) / 1_000_000


def _tokens_cost(model: str, tokens: int, table: dict[str, dict[str, float]]) -> float:
    p = table.get(model) or table.get("_default") or {}
    return tokens * p.get("input", 0.0) / 1_000_000


# ── Trace 组装 ─────────────────────────────────────────────

def _hyde_part(hyde: HydeResult | None) -> dict | None:
    if hyde is None:
        return None
    return {
        "enabled": hyde.enabled,
        "used": hyde.used,
        "output": hyde.output,
        "error": hyde.error,
        "usage": hyde.usage,
        "latency_ms": round(hyde.latency_ms, 1),
    }


def _hits_part(hits) -> dict:
    return {"n": len(hits), "ids": [h.chunk_uid for h in hits], "scores": [h.score for h in hits]}


def build_trace(
    *,
    question: str,
    config_id: str,
    hyde: HydeResult | None = None,
    retrieval: RetrievalResult | None = None,
    generation: GenerationResult | None = None,
    prompt_version: str = "",
    index_meta: dict | None = None,
    latency_total_ms: float = 0.0,
) -> dict[str, Any]:
    pricing = load_pricing_config()
    peak = is_peak_time()
    s = get_settings()

    trace: dict[str, Any] = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "trace_id": new_trace_id(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "config_id": config_id,
        "hyde": _hyde_part(hyde),
        "vector": None,
        "bm25": None,
        "rrf": None,
        "rerank": None,
        "context": None,
        "generation": None,
        "totals": {},
        "versions": {
            "commit": git_commit(),
            "prompt": prompt_version,
            "gen_model": s.gen_model,
            "embed_model": s.embed_api_model if s.embed_mode == "api" else s.embed_local_model,
            "rerank_model": s.rerank_api_model if s.rerank_mode == "api" else s.rerank_local_model,
            "index": index_meta or {},
            "schema": TRACE_SCHEMA_VERSION,
        },
    }

    cost_by_stage: dict[str, float] = {"hyde": 0.0, "rerank": 0.0, "generation": 0.0}

    if retrieval is not None:
        rt = retrieval.trace
        trace["vector"] = _hits_part(rt.vector_hits)
        trace["bm25"] = _hits_part(rt.bm25_hits)
        trace["rrf"] = rt.fusion
        trace["rerank"] = rt.rerank
        trace["context"] = {"parent_ids": rt.context_parent_ids, "chars": rt.context_chars}

    if hyde is not None and hyde.usage:
        cost_by_stage["hyde"] = _gen_cost(s.gen_model, hyde.usage, pricing, peak)

    if retrieval is not None:
        rerank_usage = (retrieval.trace.rerank or {}).get("usage") or {}
        cost_by_stage["rerank"] = _tokens_cost(s.rerank_api_model, int(rerank_usage.get("total_tokens", 0) or 0), pricing.rerank)

    if generation is not None:
        trace["generation"] = {
            "answer": generation.answer,
            "citations": [asdict(c) for c in generation.citations],
            "citation_stats": generation.citation_stats,
            "model": generation.model,
            "prompt_version": generation.prompt_version,
            "usage": generation.usage,
            "latency_ms": round(generation.latency_ms, 1),
        }
        cost_by_stage["generation"] = _gen_cost(s.gen_model, generation.usage, pricing, peak)

    tokens = {
        "hyde": (hyde.usage or {}).get("total_tokens", 0) if hyde else 0,
        "generation": (generation.usage or {}).get("total_tokens", 0) if generation else 0,
    }
    trace["totals"] = {
        "tokens": tokens,
        "cost_by_stage_cny": {k: round(v, 6) for k, v in cost_by_stage.items()},
        "cost_total_cny": round(sum(cost_by_stage.values()), 6),
        "cost_estimated": True,
        "latency_ms": {
            "hyde": round(hyde.latency_ms, 1) if hyde else 0.0,
            "retrieval": retrieval.trace.latency_ms if retrieval else 0.0,
            "generation": round(generation.latency_ms, 1) if generation else 0.0,
            "total": round(latency_total_ms, 1),
        },
    }
    return trace


class TraceWriter:
    """JSONL 落盘：一行一条 trace（P1 评测 runner 的标准落盘口）。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, trace: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
