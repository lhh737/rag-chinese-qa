"""融合（RRF / merge）与动态 TopK 测试。"""
from __future__ import annotations

import pytest

from ragqa.config.loader import RetrievalConfig
from ragqa.retrieval.hybrid import HybridRetriever, dynamic_parent_k
from ragqa.types import Document


def _doc(uid: str) -> Document:
    return Document(page_content=f"内容 {uid}", metadata={"chunk_uid": uid})


def _retriever(fusion: str = "rrf", rrf_k: int = 60) -> HybridRetriever:
    return HybridRetriever(store=None, cfg=RetrievalConfig(fusion=fusion, rrf_k=rrf_k))  # _fuse 不触碰 store


def test_rrf_score_and_order():
    r = _retriever()
    vector = [(_doc("a"), 0.1), (_doc("b"), 0.2), (_doc("c"), 0.3)]
    bm25 = [(_doc("c"), 9.0), (_doc("a"), 5.0)]
    pool, info = r._fuse(vector, bm25)

    # a: 1/(60+1) + 1/(60+2)；c: 1/(60+3) + 1/(60+1)；b: 1/(60+2)
    sa = 1 / 61 + 1 / 62
    sc = 1 / 63 + 1 / 61
    sb = 1 / 62
    assert [d.metadata["chunk_uid"] for d in pool] == ["a", "c", "b"]
    assert info["strategy"] == "rrf" and info["k"] == 60
    assert info["fused"][0]["rrf_score"] == pytest.approx(sa, abs=1e-6)
    assert info["fused"][1]["rrf_score"] == pytest.approx(sc, abs=1e-6)
    assert info["fused"][2]["rrf_score"] == pytest.approx(sb, abs=1e-6)
    # 明细记录两路排名
    assert info["fused"][0]["vector_rank"] == 1 and info["fused"][0]["bm25_rank"] == 2


def test_rrf_dedup_by_uid():
    r = _retriever()
    vector = [(_doc("x"), 0.1), (_doc("x"), 0.2)]  # 重复 uid（理论不应出现，防御性）
    pool, info = r._fuse(vector, [])
    assert len(pool) == 1 and info["n_fused"] == 1


def test_rrf_k_changes_scores():
    low = _retriever(rrf_k=10)._fuse([(_doc("a"), 1), (_doc("b"), 2)], [])[1]
    high = _retriever(rrf_k=100)._fuse([(_doc("a"), 1), (_doc("b"), 2)], [])[1]
    assert low["fused"][0]["rrf_score"] > high["fused"][0]["rrf_score"]  # 小 k 放大排名差异


def test_merge_strategy_old_behavior():
    r = _retriever(fusion="merge")
    vector = [(_doc("a"), 0.1), (_doc("b"), 0.2)]
    bm25 = [(_doc("c"), 9.0), (_doc("a"), 5.0)]
    pool, info = r._fuse(vector, bm25)
    # 旧行为：向量序优先，新块追加在后
    assert [d.metadata["chunk_uid"] for d in pool] == ["a", "b", "c"]
    assert info["strategy"] == "merge"


def test_dynamic_parent_k_rules():
    long_q = "很长的问题" * 10  # ≥36 字符 → +2
    spread12 = [1.0 - i * 0.05 for i in range(12)]          # 12 个候选、分数差异大
    dense12 = [0.50 - i * 0.001 for i in range(12)]          # 12 个候选、分数密集（<0.12）
    assert dynamic_parent_k(long_q, spread12, base=4, max_k=8) == 6   # 长问题 +2
    assert dynamic_parent_k("短问题", dense12, base=4, max_k=8) == 5  # 分数密集 +1
    assert dynamic_parent_k("短问题", spread12, base=4, max_k=8) == 4
    assert dynamic_parent_k("短问题", [0.9, 0.1], base=4, max_k=8) == 4  # 下限为 base
    assert dynamic_parent_k("短问题", [], base=4, max_k=8) == 4
