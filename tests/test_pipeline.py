"""数据管道测试：幂等、清洗生效、持久化一致性（离线，假嵌入）。"""
from __future__ import annotations

from ragqa.ingestion.pipeline import ingest_paths, verify_store
from ragqa.retrieval.store import VectorStore

ZWSP = chr(0x200B)


def _mk_store(tmp_path, fake_embed) -> VectorStore:
    return VectorStore(persist_dir=tmp_path / "idx", embed_model=fake_embed, auto_load=False)


def test_ingest_idempotent_and_sanitize(tmp_path, fake_embed):
    store = _mk_store(tmp_path, fake_embed)
    src = tmp_path / "doc1.txt"
    src.write_text("这是一段用于测试的数据管道文档。" * 60 + ZWSP, encoding="utf-8")

    r1 = ingest_paths([src], store=store)
    assert r1[0].status == "indexed"
    assert r1[0].chars_removed == 1          # 零宽字符被剥离
    assert r1[0].n_children > 0

    r2 = ingest_paths([src], store=store)    # 幂等：内容未变 → 跳过
    assert r2[0].status == "skipped"

    v = verify_store(store)
    assert v["ok"] and v["n_docs"] == 1 and v["n_chunks"] == store.count


def test_ingest_persist_reload(tmp_path, fake_embed):
    store = _mk_store(tmp_path, fake_embed)
    src = tmp_path / "doc2.txt"
    src.write_text("第二篇测试文档内容。" * 80, encoding="utf-8")
    ingest_paths([src], store=store)

    reloaded = VectorStore(persist_dir=tmp_path / "idx", embed_model=fake_embed)
    assert reloaded.count == store.count
    assert reloaded.has_document("doc2")


def test_ingest_failure_recorded(tmp_path, fake_embed):
    store = _mk_store(tmp_path, fake_embed)
    bad = tmp_path / "bad.txt"
    bad.write_text("", encoding="utf-8")  # 空文档 → 解析出 0 块
    r = ingest_paths([bad], store=store)
    assert r[0].status == "failed" and "文本块" in (r[0].error or "")
