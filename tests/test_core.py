"""门面端到端测试（离线，假嵌入/假重排/假生成）：query 与 astream。"""
from __future__ import annotations

import asyncio

from conftest import FakeAsyncGenClient, FakeGenClient
from ragqa.config.loader import HydeConfig, RetrievalConfig
from ragqa.core import RAGPipeline
from ragqa.retrieval.store import VectorStore
from ragqa.types import Document


def _prep_store(tmp_path, fake_embed) -> VectorStore:
    store = VectorStore(persist_dir=tmp_path / "idx", embed_model=fake_embed, auto_load=False)
    text = "Transformer 使用自注意力机制，计算复杂度为 O(n²)，与序列长度平方相关。"
    parents = [Document(page_content=text, metadata={"parent_id": "1706.03762:p0000", "doc_id": "1706.03762", "source": "1706.03762.pdf"})]
    children = [Document(page_content=text, metadata={"chunk_uid": "1706.03762:c00000", "parent_id": "1706.03762:p0000", "doc_id": "1706.03762", "source": "1706.03762.pdf"})]
    store.add_document(
        parents, children,
        {"doc_id": "1706.03762", "content_hash": "h1", "path": "", "filename": "1706.03762.pdf", "status": "indexed", "loaded_at": "2026-10-05"},
    )
    return store


def _cfg() -> RetrievalConfig:
    return RetrievalConfig(hyde=HydeConfig(enabled=False))


def test_query_end_to_end(tmp_path, fake_embed, fake_reranker):
    store = _prep_store(tmp_path, fake_embed)
    pipe = RAGPipeline(
        retrieval_config=_cfg(), store=store, reranker=fake_reranker,
        gen_client=FakeGenClient("注意力机制的复杂度为 O(n²) [1]。"),
    )
    res = pipe.query("注意力机制的复杂度是多少？")

    assert res.answer.startswith("注意力机制")
    assert res.contexts and res.contexts[0].doc_id == "1706.03762"
    assert res.citations and res.citations[0].valid and res.citations[0].source_doc == "1706.03762"
    trace = res.trace
    assert trace["rrf"]["strategy"] == "rrf"
    assert trace["context"]["parent_ids"] == ["1706.03762:p0000"]
    assert trace["generation"]["citation_stats"]["n_markers"] == 1
    assert trace["versions"]["index"]["count"] == 1


def test_query_prompt_version_passed(tmp_path, fake_embed, fake_reranker):
    store = _prep_store(tmp_path, fake_embed)
    client = FakeGenClient("答案 [1]。")
    pipe = RAGPipeline(retrieval_config=_cfg(), store=store, reranker=fake_reranker, gen_client=client)
    pipe.query("问题", prompt_version="v1")
    assert client.calls and client.calls[0]["messages"][0]["role"] == "system"
    assert "【参考资料1】" in client.calls[0]["messages"][0]["content"]


def test_query_empty_store(tmp_path, fake_embed):
    store = VectorStore(persist_dir=tmp_path / "empty", embed_model=fake_embed, auto_load=False)
    pipe = RAGPipeline(retrieval_config=_cfg(), store=store)
    res = pipe.query("问题")
    assert "知识库为空" in res.answer
    assert res.trace["schema_version"] == 1


def test_astream_end_to_end(tmp_path, fake_embed, fake_reranker):
    store = _prep_store(tmp_path, fake_embed)
    pipe = RAGPipeline(retrieval_config=_cfg(), store=store, reranker=fake_reranker)
    pipe._async_gen_client = FakeAsyncGenClient(["答案：", "复杂度 O(n²) ", "[1]。"])

    async def run():
        chunks = []
        async for ch in pipe.astream("给我答案"):
            chunks.append(ch)
        return chunks

    chunks = asyncio.run(run())
    assert not chunks[-1].done or True  # 最后一个块必须 done
    assert chunks[-1].done
    result = chunks[-1].result
    assert result is not None
    assert result.answer == "答案：复杂度 O(n²) [1]。"
    assert result.citations and result.citations[0].valid
    assert "".join(c.delta for c in chunks if not c.done) == "答案：复杂度 O(n²) [1]。"
