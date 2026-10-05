"""
BM25 + 向量混合检索，Cross-Encoder 重排序，TopK 动态调节。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jieba
import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from model.factory import get_reranker
from utils.config_handler import faiss_conf
from utils.logger_handler import logger

if TYPE_CHECKING:
    from rag.vector_store import VectorStoreService


def _zh_tokenize(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    try:
        return [t for t in jieba.cut(text) if t.strip()]
    except Exception:
        return list(text)


def _dynamic_parent_k(query: str, rerank_scores: list[float], base: int, max_k: int) -> int:
    k = base
    if len(query) >= 36:
        k += 2
    if len(rerank_scores) >= 4:
        top = rerank_scores[: min(12, len(rerank_scores))]
        if max(top) - min(top) < 0.12:
            k += 1
    return max(base, min(k, max_k, len(rerank_scores) if rerank_scores else base))


class HybridRetriever:
    def __init__(self, vs: "VectorStoreService"):
        self.vs = vs

    def _bm25_candidates(self, query: str, tokenized_corpus: list[list[str]], child_docs: list[Document], k: int):
        if not tokenized_corpus or not child_docs:
            return []
        bm25 = BM25Okapi(tokenized_corpus)
        q_tokens = _zh_tokenize(query)
        if not q_tokens:
            return []
        scores = bm25.get_scores(q_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [child_docs[i] for i in ranked]

    def retrieve_for_rag(self, query: str) -> tuple[str, list[dict[str, Any]]]:
        """
        返回：（拼接后的父文档上下文，溯源列表）。
        溯源项：source 文件名、parent_id、摘录。
        """
        vs = self.vs
        if vs.faiss_store is None or not vs.child_documents:
            return "", []

        vector_k = int(faiss_conf.get("vector_fetch_k", 20))
        bm25_k = int(faiss_conf.get("bm25_fetch_k", 20))
        rerank_n = int(faiss_conf.get("rerank_top_n", 12))
        base_k = int(faiss_conf.get("base_context_k", 4))
        max_k = int(faiss_conf.get("max_context_k", 8))

        vec_docs = vs.faiss_store.similarity_search(query, k=vector_k)
        bm_docs = self._bm25_candidates(query, vs.tokenized_corpus, vs.child_documents, bm25_k)

        merged: dict[Any, Document] = {}
        for d in vec_docs:
            uid = d.metadata.get("chunk_uid")
            key = uid if uid is not None else id(d)
            merged[key] = d
        for d in bm_docs:
            uid = d.metadata.get("chunk_uid")
            key = uid if uid is not None else id(d)
            merged.setdefault(key, d)
        pool = list(merged.values())
        if not pool:
            return "", []

        reranker = get_reranker()
        if reranker is not None:
            try:
                # 本地 CrossEncoder: predict() 返回分数列表
                if hasattr(reranker, "predict"):
                    pairs = [[query, d.page_content] for d in pool]
                    raw = reranker.predict(pairs, show_progress_bar=False)
                    scores = np.asarray(raw).reshape(-1).tolist()
                # DashScope API: rerank() 返回 [{"index": int, "relevance_score": float}, ...]
                elif hasattr(reranker, "rerank"):
                    docs_text = [d.page_content for d in pool]
                    results = reranker.rerank(query, docs_text, top_n=len(pool))
                    scores = [0.0] * len(pool)
                    for r in results:
                        scores[r["index"]] = r.get("relevance_score", 0.0)
                else:
                    raise ValueError("未知的 reranker 类型")
            except Exception as e:
                logger.warning("[HybridRetriever] rerank 失败，回退向量顺序: %s", e)
                scores = list(range(len(pool), 0, -1))
        else:
            scores = list(range(len(pool), 0, -1))

        ranked = sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)[:rerank_n]
        rerank_scores = [float(s) for _, s in ranked]
        take_parents = _dynamic_parent_k(query, rerank_scores, base_k, max_k)

        seen_parent: set[str] = set()
        ordered_parents: list[tuple[str, str, dict]] = []
        provenance: list[dict[str, Any]] = []

        for doc, _score in ranked:
            pid = doc.metadata.get("parent_id")
            if not pid or pid in seen_parent:
                continue
            seen_parent.add(pid)
            blob = vs.parent_store.get(pid)
            if not blob:
                continue
            text = blob.get("page_content", "")
            meta = blob.get("metadata") or {}
            source = meta.get("source") or doc.metadata.get("source", "")
            ordered_parents.append((pid, text, meta))
            provenance.append(
                {
                    "parent_id": pid,
                    "source": source,
                    "excerpt": text[:320].replace("\n", " ") + ("…" if len(text) > 320 else ""),
                }
            )
            if len(ordered_parents) >= take_parents:
                break

        if not ordered_parents:
            return "", []

        context_parts = []
        for i, (_pid, text, _meta) in enumerate(ordered_parents, start=1):
            context_parts.append(f"【参考资料{i}】\n{text}")
        return "\n\n".join(context_parts), provenance
