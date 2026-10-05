"""向量存储测试：假嵌入离线验证 增→存→载→查 往返与一致性校验。"""
from __future__ import annotations

import json

import pytest

from ragqa.retrieval.store import StoreInconsistencyError, VectorStore
from ragqa.types import Document

from conftest import FakeEmbedding


def _mk_store(tmp_path, embed, auto_load=False) -> VectorStore:
    return VectorStore(persist_dir=tmp_path / "idx", embed_model=embed, auto_load=auto_load)


def _add_doc(store: VectorStore, doc_id: str = "d1", texts: list[str] | None = None) -> bool:
    texts = texts or ["Transformer 使用自注意力机制。", "注意力机制计算复杂度是 O(n²)。", "今天是晴天。"]
    parents = [Document(page_content="父块内容。" * 50, metadata={"parent_id": f"{doc_id}:p0000", "doc_id": doc_id, "source": f"{doc_id}.pdf"})]
    children = [
        Document(page_content=t, metadata={"chunk_uid": f"{doc_id}:c{i:05d}", "parent_id": f"{doc_id}:p0000", "doc_id": doc_id, "source": f"{doc_id}.pdf"})
        for i, t in enumerate(texts)
    ]
    info = {"doc_id": doc_id, "content_hash": f"hash-{doc_id}", "path": "", "filename": f"{doc_id}.pdf", "status": "indexed", "loaded_at": "2026-10-05"}
    return store.add_document(parents, children, info)


def test_add_save_reload_search(tmp_path, fake_embed):
    store = _mk_store(tmp_path, fake_embed)
    assert _add_doc(store)
    store.save()

    s2 = _mk_store(tmp_path, fake_embed, auto_load=True)
    assert s2.count == 3 and not s2.is_empty

    hits = s2.search_with_scores("Transformer 使用自注意力机制。", k=2)
    assert hits and hits[0][0].page_content == "Transformer 使用自注意力机制。"
    top_bm25 = s2.bm25_rank("晴天", k=1)
    assert top_bm25 and "晴天" in top_bm25[0][0].page_content
    assert s2.get_parent("d1:p0000") is not None


def test_idempotent_add(tmp_path, fake_embed):
    store = _mk_store(tmp_path, fake_embed)
    assert _add_doc(store) is True
    assert store.has_document("d1", "hash-d1")
    assert _add_doc(store) is False  # 同 doc 同 hash → 跳过


def test_row_alignment_after_reload(tmp_path, fake_embed):
    store = _mk_store(tmp_path, fake_embed)
    _add_doc(store, "d1", ["第一篇文章的内容甲。", "第一篇文章的内容乙。"])
    _add_doc(store, "d2", ["第二篇文章的内容丙。"])
    store.save()

    s2 = _mk_store(tmp_path, fake_embed, auto_load=True)
    hits = s2.search_with_scores("第二篇文章的内容丙", k=1)
    assert hits[0][0].metadata["doc_id"] == "d2"  # 行号映射未错位


def test_count_mismatch_raises(tmp_path, fake_embed):
    store = _mk_store(tmp_path, fake_embed)
    _add_doc(store)
    store.save()
    # 篡改：删掉一个子块 → 与索引行数不一致，必须显式失败
    p = tmp_path / "idx" / "child_docs.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    p.write_text(json.dumps(data[:-1], ensure_ascii=False), encoding="utf-8")
    with pytest.raises(StoreInconsistencyError):
        VectorStore(persist_dir=tmp_path / "idx", embed_model=fake_embed)


def test_meta_dim_recorded(tmp_path, fake_embed):
    store = _mk_store(tmp_path, fake_embed)
    _add_doc(store)
    store.save()
    assert store.meta["dim"] == FakeEmbedding.dim
    assert store.meta["count"] == 3
