"""FastAPI 服务层（P7）：与 CLI / Gradio / 评测引擎共用同一入口（core.RAGPipeline）。

运行：`uv run ragqa serve [--host H --port P]`（uvicorn factory 模式——导入期不加载索引，
便于测试注入 stub pipeline）。

端点：
- GET  /health     服务与索引元信息（文档数 / 维度 / 嵌入模型 / 检索配置）
- GET  /documents  已索引文档清单
- POST /query      {"question": "...", "prompt_version": "v1"?} → 答案 + 引用 + 上下文数 + 延迟
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ragqa.core import RAGPipeline


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    prompt_version: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    n_contexts: int
    latency_ms: float
    config_id: str


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    """应用工厂。`pipeline=None` 时使用真实管道（延迟到首次请求前创建）。"""
    app = FastAPI(title="rag-chinese-qa", version="2.0.0")
    app.state.pipeline = pipeline

    def _pipeline() -> RAGPipeline:
        if app.state.pipeline is None:
            from ragqa.retrieval.store import VectorStore

            app.state.pipeline = RAGPipeline(store=VectorStore())
        return app.state.pipeline

    @app.get("/health")
    def health() -> dict[str, Any]:
        p = _pipeline()
        meta = getattr(p.store, "meta", {}) or {}
        try:
            n_docs = len(p.get_file_list())
        except Exception:  # noqa: BLE001
            n_docs = None
        return {
            "status": "ok",
            "index": {
                "n_docs": n_docs,
                "n_chunks": meta.get("count"),
                "dim": meta.get("dim"),
                "embed_model": meta.get("embed_model"),
                "pipeline_version": meta.get("pipeline_version"),
            },
            "config_id": p.retrieval_cfg.config_id(),
            "gen_prompt": p.generation_cfg.prompt_version,
        }

    @app.get("/documents")
    def documents() -> dict[str, Any]:
        return {"files": _pipeline().get_file_list()}

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        p = _pipeline()
        if p.store.is_empty:
            raise HTTPException(status_code=503, detail="知识库为空，请先导入文档")
        t0 = time.perf_counter()
        r = p.query(req.question, prompt_version=req.prompt_version)
        return QueryResponse(
            answer=r.answer,
            citations=[
                {"answer_sentence": c.answer_sentence, "marker": c.marker, "valid": c.valid,
                 "source_doc": c.source_doc}
                for c in r.citations
            ],
            n_contexts=len(r.contexts),
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            config_id=str((r.trace or {}).get("config_id") or p.retrieval_cfg.config_id()),
        )

    return app
