"""统一查询门面（设计 §5.3）：评测引擎 / FastAPI / Gradio / 评审查看器共用同一入口。

    pipeline = RAGPipeline()
    result = pipeline.query("问题")
    result.answer / result.contexts / result.citations / result.trace

评测对象与线上行为一致：所有入口都走这里。
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator

from ragqa.config.loader import (
    GenerationConfig,
    RetrievalConfig,
    load_generation_config,
    load_retrieval_config,
)
from ragqa.generation.generator import GenerationSession, generate
from ragqa.retrieval.hybrid import HybridRetriever
from ragqa.retrieval.hyde import generate_hypothetical_doc
from ragqa.retrieval.store import VectorStore
from ragqa.trace import build_trace
from ragqa.types import QueryResult, StreamChunk
from ragqa.utils.logging import logger


class RAGPipeline:
    def __init__(
        self,
        *,
        retrieval_config: RetrievalConfig | None = None,
        generation_config: GenerationConfig | None = None,
        store: VectorStore | None = None,
        reranker: Any | None = None,
        gen_client: Any | None = None,
    ) -> None:
        self.retrieval_cfg = retrieval_config or load_retrieval_config()
        self.generation_cfg = generation_config or load_generation_config()
        self.store = store or VectorStore()
        self.retriever = HybridRetriever(self.store, self.retrieval_cfg, reranker)
        self._gen_client = gen_client  # 测试可注入；None 时工厂创建
        self._async_gen_client = None  # 流式客户端（懒创建）

    # ── 文档管理（UI 用）─────────────────────────────────

    def get_file_list(self) -> list[dict]:
        return self.store.file_list

    def get_document_content(self, doc_id: str) -> str | None:
        return self.store.get_document_content(doc_id)

    # ── 检索（HyDE → 混合检索）──────────────────────────

    def _retrieve(self, question: str, cfg: RetrievalConfig):
        hyde = generate_hypothetical_doc(question, cfg.hyde, client=self._gen_client)
        retrieval = self.retriever.retrieve(hyde.query)
        return retrieval, hyde

    def _empty_answer(self, question: str, cfg: RetrievalConfig, msg: str, t0: float) -> QueryResult:
        trace = build_trace(
            question=question,
            config_id=cfg.config_id(),
            prompt_version=self.generation_cfg.prompt_version,
            index_meta=self.store.meta,
            latency_total_ms=(time.perf_counter() - t0) * 1000,
        )
        return QueryResult(answer=msg, contexts=[], citations=[], trace=trace)

    # ── 查询 ─────────────────────────────────────────────

    def query(
        self,
        question: str,
        *,
        config: RetrievalConfig | None = None,
        prompt_version: str | None = None,
    ) -> QueryResult:
        cfg = config or self.retrieval_cfg
        t0 = time.perf_counter()

        if self.store.is_empty:
            return self._empty_answer(question, cfg, "知识库为空，请先导入文档。", t0)

        retrieval, hyde = self._retrieve(question, cfg)
        generation = None
        if retrieval.blocks:
            generation = generate(
                question,
                retrieval.context,
                retrieval.blocks,
                prompt_version=prompt_version,
                cfg=self.generation_cfg,
                client=self._gen_client,
            )
            answer = generation.answer
        else:
            answer = "未找到相关文档内容。"
            logger.info("[core] 无检索结果: %s", question)

        trace = build_trace(
            question=question,
            config_id=cfg.config_id(),
            hyde=hyde,
            retrieval=retrieval,
            generation=generation,
            prompt_version=prompt_version or self.generation_cfg.prompt_version,
            index_meta=self.store.meta,
            latency_total_ms=(time.perf_counter() - t0) * 1000,
        )
        return QueryResult(
            answer=answer,
            contexts=retrieval.blocks,
            citations=generation.citations if generation else [],
            trace=trace,
        )

    async def astream(
        self,
        question: str,
        *,
        config: RetrievalConfig | None = None,
        prompt_version: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """流式查询：增量 yield delta；最后一个块 done=True 且携带完整 QueryResult。"""
        cfg = config or self.retrieval_cfg
        t0 = time.perf_counter()

        if self.store.is_empty:
            yield StreamChunk(done=True, result=self._empty_answer(question, cfg, "知识库为空，请先导入文档。", t0))
            return

        # HyDE 与检索是阻塞调用，放到线程池避免卡住事件循环
        retrieval, hyde = await asyncio.to_thread(self._retrieve, question, cfg)
        if not retrieval.blocks:
            yield StreamChunk(done=True, result=self._empty_answer(question, cfg, "未找到相关文档内容。", t0))
            return

        session = GenerationSession(
            question,
            retrieval.context,
            retrieval.blocks,
            prompt_version=prompt_version,
            cfg=self.generation_cfg,
            client=self._async_gen_client,
        )
        async for delta in session:
            yield StreamChunk(delta=delta)

        generation = session.result
        trace = build_trace(
            question=question,
            config_id=cfg.config_id(),
            hyde=hyde,
            retrieval=retrieval,
            generation=generation,
            prompt_version=prompt_version or self.generation_cfg.prompt_version,
            index_meta=self.store.meta,
            latency_total_ms=(time.perf_counter() - t0) * 1000,
        )
        yield StreamChunk(
            done=True,
            result=QueryResult(
                answer=generation.answer if generation else "",
                contexts=retrieval.blocks,
                citations=generation.citations if generation else [],
                trace=trace,
            ),
        )
