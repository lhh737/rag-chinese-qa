"""FastAPI 服务层（P7）：与 CLI / Web UI / MCP server / 评测引擎共用同一入口（core.RAGPipeline）。

运行：`uv run ragqa serve [--host H --port P]` → 打开 http://127.0.0.1:8000 （Web UI）

端点：
- GET  /                     Web UI（apps/web_ui.html，单文件原生前端）
- GET  /health               服务与索引元信息（文档数 / 维度 / 嵌入模型 / 检索配置）
- GET  /documents            已索引文档清单
- GET  /documents/{id}/content  文档原文（查看用）
- POST /documents/upload     上传文档并入库（multipart，幂等：内容未变则跳过）
- POST /documents/import_papers  批量导入论文库 + 内置中文文档
- POST /query                {"question": "..."} → 答案 + 引用 + 延迟（非流式，评测同源）
- POST /query/stream         同 /query，SSE 流式：retrieval → delta* → done（UI 主链路）
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ragqa.config.settings import get_settings
from ragqa.core import RAGPipeline
from ragqa.ingestion.pipeline import SUPPORTED_EXTS, ingest_paths
from ragqa.types import ContextBlock, QueryResult, StreamChunk
from ragqa.utils.paths import REPO_ROOT, resolve_path

WEB_UI_PATH: Path = REPO_ROOT / "apps" / "web_ui.html"


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    prompt_version: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    n_contexts: int
    latency_ms: float
    config_id: str


# ── 序列化（SSE 载荷）─────────────────────────────────────

def _block(b: ContextBlock) -> dict[str, Any]:
    return {
        "rank": b.rank,
        "doc_id": b.doc_id,
        "source": b.source,
        "text": b.text,
        "rerank_score": b.rerank_score,
    }


def _retrieval_payload(chunk: StreamChunk) -> dict[str, Any]:
    """检索完成事件：来源块 + 漏斗计数（向量/BM25 → 融合 → 重排 → 上下文）。"""
    ret = chunk.retrieval
    assert ret is not None
    rt = ret.trace
    fusion = rt.fusion or {}
    rerank = rt.rerank or {}
    hyde = chunk.hyde
    return {
        "funnel": {
            "strategy": fusion.get("strategy"),
            "rrf_k": fusion.get("k"),
            "n_vector": fusion.get("n_vector"),
            "n_bm25": fusion.get("n_bm25"),
            "n_fused": fusion.get("n_fused"),
            "n_reranked": len(rerank.get("scores") or []),
            "n_context": len(ret.blocks),
        },
        "hyde": {"enabled": hyde.enabled, "used": hyde.used} if hyde else None,
        "rerank": {"latency_ms": rerank.get("latency_ms"), "error": rerank.get("error")},
        "latency_ms": rt.latency_ms,
        "contexts": [_block(b) for b in ret.blocks],
    }


def _done_payload(r: QueryResult) -> dict[str, Any]:
    """完成事件：最终答案 + 逐句引用 + trace 摘要（延迟分解 / 成本 / 版本指纹）。"""
    t = r.trace or {}
    totals = t.get("totals") or {}
    versions = t.get("versions") or {}
    gen = t.get("generation") or {}
    return {
        "answer": r.answer,
        "citations": [
            {"answer_sentence": c.answer_sentence, "marker": c.marker, "valid": c.valid,
             "source_doc": c.source_doc}
            for c in r.citations
        ],
        "citation_stats": gen.get("citation_stats") or {},
        "trace": {
            "trace_id": t.get("trace_id"),
            "config_id": t.get("config_id"),
            "prompt_version": versions.get("prompt") or gen.get("prompt_version"),
            "gen_model": versions.get("gen_model") or gen.get("model"),
            "latency_ms": totals.get("latency_ms") or {},
            "tokens": totals.get("tokens") or {},
            "usage": gen.get("usage") or {},          # 生成阶段 usage（include_usage 回传时非空）
            "cost_cny": totals.get("cost_total_cny"),
            "cost_by_stage_cny": totals.get("cost_by_stage_cny") or {},
            "cost_estimated": totals.get("cost_estimated", True),
        },
    }


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ── 应用工厂 ──────────────────────────────────────────────

def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    """应用工厂。`pipeline=None` 时使用真实管道（延迟到首次请求前创建）。"""
    app = FastAPI(title="rag-chinese-qa", version="2.0.0")
    app.state.pipeline = pipeline

    def _pipeline() -> RAGPipeline:
        if app.state.pipeline is None:
            from ragqa.retrieval.store import VectorStore

            app.state.pipeline = RAGPipeline(store=VectorStore())
        return app.state.pipeline

    # ── Web UI ───────────────────────────────────────────

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        if not WEB_UI_PATH.is_file():
            raise HTTPException(status_code=500, detail="apps/web_ui.html 缺失")
        return FileResponse(WEB_UI_PATH, media_type="text/html")

    # ── 服务与索引 ────────────────────────────────────────

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

    @app.get("/documents/{doc_id}/content")
    def document_content(doc_id: str) -> dict[str, Any]:
        content = _pipeline().get_document_content(doc_id)
        if not content:
            raise HTTPException(status_code=404, detail=f"文档不存在或无法读取: {doc_id}")
        return {"doc_id": doc_id, "content": content}

    # ── 文档管理（同步 def：FastAPI 自动走线程池，不阻塞事件循环）──

    @app.post("/documents/upload")
    def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
        name = os.path.basename(file.filename or "")
        if not name or Path(name).suffix.lower() not in SUPPORTED_EXTS:
            raise HTTPException(status_code=400, detail=f"仅支持 {sorted(SUPPORTED_EXTS)} 文档")
        dest_dir = get_settings().upload_dir_path
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / name
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)

        p = _pipeline()
        results = ingest_paths([dest], store=p.store)
        r = results[0]
        return {
            "doc_id": r.doc_id,
            "status": r.status,          # indexed | skipped | failed
            "n_parents": r.n_parents,
            "n_children": r.n_children,
            "error": r.error,
            "n_docs": len(p.get_file_list()),
        }

    @app.post("/documents/import_papers")
    def import_papers() -> dict[str, Any]:
        s = _pipeline()
        paths: list[Path] = []
        papers = get_settings().papers_dir_path
        if papers.is_dir():
            paths.append(papers)
        for f in ("data/llm_intro.txt", "data/test_transformer.txt"):
            fp = resolve_path(f)
            if fp.is_file():
                paths.append(fp)
        if not paths:
            raise HTTPException(status_code=404, detail="论文目录不存在（data/papers）")

        results = ingest_paths(paths, store=s.store)
        return {
            "indexed": sum(1 for r in results if r.status == "indexed"),
            "skipped": sum(1 for r in results if r.status == "skipped"),
            "failed": sum(1 for r in results if r.status == "failed"),
            "n_files": len(results),
            "n_docs": len(s.get_file_list()),
        }

    # ── 查询 ─────────────────────────────────────────────

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        p = _pipeline()
        if p.store.is_empty:
            raise HTTPException(status_code=503, detail="知识库为空，请先导入文档")
        r = p.query(req.question, prompt_version=req.prompt_version)
        return QueryResponse(
            answer=r.answer,
            citations=[
                {"answer_sentence": c.answer_sentence, "marker": c.marker, "valid": c.valid,
                 "source_doc": c.source_doc}
                for c in r.citations
            ],
            n_contexts=len(r.contexts),
            latency_ms=round(float((r.trace.get("totals") or {}).get("latency_ms", {}).get("total", 0.0)), 1),
            config_id=str((r.trace or {}).get("config_id") or p.retrieval_cfg.config_id()),
        )

    @app.post("/query/stream")
    async def query_stream(req: QueryRequest) -> StreamingResponse:
        p = _pipeline()
        if p.store.is_empty:
            raise HTTPException(status_code=503, detail="知识库为空，请先导入文档")

        async def gen() -> AsyncIterator[str]:
            try:
                async for ch in p.astream(req.question, prompt_version=req.prompt_version):
                    if ch.stage == "retrieval":
                        yield _sse("retrieval", _retrieval_payload(ch))
                    elif ch.done and ch.result is not None:
                        yield _sse("done", _done_payload(ch.result))
                    elif ch.delta:
                        yield _sse("delta", {"text": ch.delta})
            except Exception as e:  # noqa: BLE001 —— 流中途出错以事件形式告知前端
                yield _sse("error", {"message": str(e)})

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
