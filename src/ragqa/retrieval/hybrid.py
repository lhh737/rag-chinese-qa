"""混合检索：向量 + BM25 双路召回 → 融合 → 重排 → 父子块还原 → 动态 TopK（设计 §5.1）。

融合策略可配（对照实验的自变量）：
- rrf  ：score(d) = Σ_lists 1/(k + rank_i(d))，rank 从 1 起，k 进配置（支持扫描）
- merge：向量优先合并去重（旧栈行为，仅依赖迁移对照实验用）

全链路产出结构化 trace 字段（向量/BM25 命中、融合明细、重排分、上下文），
是评测、错误分析与归因的唯一事实来源。
"""
from __future__ import annotations

import time
from typing import Any

from ragqa.config.loader import RetrievalConfig, load_retrieval_config
from ragqa.models.rerank import RerankError
from ragqa.retrieval.store import VectorStore
from ragqa.types import ChunkHit, ContextBlock, Document, RetrievalResult, RetrievalTrace
from ragqa.utils.logging import logger


def dynamic_parent_k(query: str, rerank_scores: list[float], base: int, max_k: int) -> int:
    """动态 TopK（沿用旧栈规则，保迁移对照语义）。"""
    k = base
    if len(query) >= 36:
        k += 2
    if len(rerank_scores) >= 4:
        top = rerank_scores[: min(12, len(rerank_scores))]
        if max(top) - min(top) < 0.12:
            k += 1
    return max(base, min(k, max_k, len(rerank_scores) if rerank_scores else base))


def format_context(blocks: list[ContextBlock]) -> str:
    """【参考资料n】编号与 block.rank 一致——答案中的 [n] 引用据此解析。"""
    return "\n\n".join(f"【参考资料{b.rank}】\n{b.text}" for b in blocks)


class HybridRetriever:
    def __init__(self, store: VectorStore, cfg: RetrievalConfig | None = None, reranker: Any | None = None) -> None:
        self.store = store
        self.cfg = cfg or load_retrieval_config()
        self._reranker = reranker

    @property
    def reranker(self):
        if self._reranker is None:
            from ragqa.models.factory import get_reranker

            self._reranker = get_reranker()
        return self._reranker

    # ── 融合 ─────────────────────────────────────────────

    def _fuse(
        self,
        vector: list[tuple[Document, float]],
        bm25: list[tuple[Document, float]],
    ) -> tuple[list[Document], dict]:
        if self.cfg.fusion == "merge":
            merged: dict[str, Document] = {}
            for d, _s in vector:
                merged.setdefault(d.metadata.get("chunk_uid", ""), d)
            for d, _s in bm25:
                merged.setdefault(d.metadata.get("chunk_uid", ""), d)
            return list(merged.values()), {
                "strategy": "merge",
                "n_vector": len(vector),
                "n_bm25": len(bm25),
                "n_fused": len(merged),
            }

        k = self.cfg.rrf_k
        scores: dict[str, float] = {}
        detail: dict[str, dict] = {}
        docs: dict[str, Document] = {}
        for name, lst in (("vector", vector), ("bm25", bm25)):
            for rank, (d, _s) in enumerate(lst, start=1):
                uid = d.metadata.get("chunk_uid", "")
                docs[uid] = d
                scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank)
                detail.setdefault(uid, {})[f"{name}_rank"] = rank
        ordered = sorted(scores, key=lambda u: scores[u], reverse=True)
        fused = [{"chunk_uid": u, "rrf_score": round(scores[u], 6), **detail[u]} for u in ordered]
        return [docs[u] for u in ordered], {
            "strategy": "rrf",
            "k": k,
            "n_vector": len(vector),
            "n_bm25": len(bm25),
            "n_fused": len(ordered),
            "fused": fused,
        }

    # ── 重排 ─────────────────────────────────────────────

    def _rerank(
        self, query: str, pool: list[Document]
    ) -> tuple[list[tuple[Document, float]], dict]:
        trace: dict[str, Any] = {"scores": [], "top": [], "latency_ms": 0.0, "usage": None, "error": None}
        if not pool:
            return [], trace
        try:
            t0 = time.time()
            results = self.reranker.rerank(query, [d.page_content for d in pool], top_n=len(pool))
            latency = (time.time() - t0) * 1000
            score_by_idx = {int(r["index"]): float(r.get("relevance_score", 0.0)) for r in results}
            scores = [score_by_idx.get(i, 0.0) for i in range(len(pool))]
            ranked = sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)[: self.cfg.rerank_top_n]
            trace.update(
                {
                    "scores": [round(float(s), 6) for _d, s in ranked],
                    "top": [d.metadata.get("chunk_uid", "") for d, _s in ranked],
                    "latency_ms": round(latency, 1),
                    "usage": getattr(self.reranker, "last_usage", None),
                }
            )
            return ranked, trace
        except RerankError as e:
            logger.warning("[hybrid] 重排失败，回退融合顺序（不阻塞主链路）: %s", e)
            trace["error"] = str(e)
            ranked = list(zip(pool, list(range(len(pool), 0, -1))))[: self.cfg.rerank_top_n]
            return ranked, trace

    # ── 主流程 ───────────────────────────────────────────

    def retrieve(self, query: str) -> RetrievalResult:
        t0 = time.time()
        trace = RetrievalTrace()
        if self.store.is_empty:
            return RetrievalResult(blocks=[], trace=trace)

        vector_raw = self.store.search_with_scores(query, self.cfg.vector_fetch_k)
        bm25_raw = self.store.bm25_rank(query, self.cfg.bm25_fetch_k)

        def _hits(raw: list[tuple[Document, float]]) -> list[ChunkHit]:
            return [
                ChunkHit(
                    chunk_uid=d.metadata.get("chunk_uid", ""),
                    score=round(float(s), 6),
                    rank=i,
                    parent_id=d.metadata.get("parent_id"),
                    doc_id=d.metadata.get("doc_id"),
                )
                for i, (d, s) in enumerate(raw, start=1)
            ]

        trace.vector_hits = _hits(vector_raw)
        trace.bm25_hits = _hits(bm25_raw)

        pool, fusion = self._fuse(vector_raw, bm25_raw)
        trace.fusion = fusion
        ranked, rerank_trace = self._rerank(query, pool)
        trace.rerank = rerank_trace

        take = dynamic_parent_k(query, [s for _d, s in ranked], self.cfg.base_context_k, self.cfg.max_context_k)
        blocks: list[ContextBlock] = []
        seen_parents: set[str] = set()
        for d, s in ranked:
            pid = d.metadata.get("parent_id")
            if not pid or pid in seen_parents:
                continue
            seen_parents.add(pid)
            blob = self.store.get_parent(pid)
            if not blob:
                continue
            meta = blob.get("metadata") or {}
            blocks.append(
                ContextBlock(
                    rank=len(blocks) + 1,
                    parent_id=pid,
                    doc_id=meta.get("doc_id") or d.metadata.get("doc_id", ""),
                    source=meta.get("source") or d.metadata.get("source", ""),
                    text=blob.get("page_content", ""),
                    rerank_score=round(float(s), 6),
                )
            )
            if len(blocks) >= take:
                break

        context = format_context(blocks)
        trace.context_parent_ids = [b.parent_id for b in blocks]
        trace.context_chars = len(context)
        trace.latency_ms = round((time.time() - t0) * 1000, 1)
        return RetrievalResult(blocks=blocks, trace=trace, context=context)
