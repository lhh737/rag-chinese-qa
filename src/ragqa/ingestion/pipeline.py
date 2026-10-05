"""最小数据管道（设计 §6，P0 版）。

状态机（轻量）：DISCOVERED → PARSED → SANITIZED → CHUNKED → EMBEDDED → INDEXED → VERIFIED
- 以 manifest 字段 + 日志体现阶段；幂等键 = doc_id + content_hash + pipeline_version
- **逐文档落盘**：任一中途中断，已入库文档不重复且可断点续跑（--retry 或重跑）
- P0 边界：索引快照 / 回滚 / SQLite manifest 留 P7；失败文档记录在返回结果与日志中
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from ragqa.config.loader import RetrievalConfig, load_retrieval_config
from ragqa.ingestion.chunker import build_parent_child
from ragqa.ingestion.loader import (
    SUPPORTED_EXTS,
    compute_content_hash,
    doc_id_of,
    load_file,
)
from ragqa.ingestion.sanitize import SanitizeStats, sanitize_text
from ragqa.retrieval.store import PIPELINE_VERSION, VectorStore
from ragqa.utils.logging import logger
from ragqa.utils.paths import REPO_ROOT


@dataclass
class IngestResult:
    doc_id: str
    status: str                     # indexed | skipped | failed
    n_parents: int = 0
    n_children: int = 0
    chars_removed: int = 0
    suspicious: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    error: str | None = None


def discover_files(paths: list[str | Path]) -> list[Path]:
    """扫描路径（文件或目录），过滤支持类型与隐藏文件，去重保序。"""
    files: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(
                sorted(
                    f
                    for f in path.rglob("*")
                    if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS and not f.name.startswith(".")
                )
            )
        elif path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            files.append(path)
        else:
            logger.warning("[ingest] 跳过（不存在或不支持）: %s", path)
    seen: set[Path] = set()
    out: list[Path] = []
    for f in files:
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def ingest_file(
    filepath: Path,
    store: VectorStore,
    cfg: RetrievalConfig,
    *,
    sanitize: bool = True,
) -> IngestResult:
    t0 = time.time()
    did = doc_id_of(filepath)
    chash = compute_content_hash(filepath)

    # PARSED
    pages = load_file(filepath)
    # SANITIZED（防御第 1 层：剥离零宽/控制符 + 可疑指令标记，不静默删除）
    stats = SanitizeStats()
    if sanitize:
        for p in pages:
            cleaned, s = sanitize_text(p.page_content)
            p.page_content = cleaned
            stats.merge(s)
    # CHUNKED
    parents, children = build_parent_child(pages, cfg)
    if not children:
        raise ValueError("未提取到文本块（解析失败或空文档）")

    info = {
        "doc_id": did,
        "content_hash": chash,
        "path": str(filepath.resolve()),
        "filename": filepath.name,
        "size_kb": round(filepath.stat().st_size / 1024, 1),
        "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "indexed",
        "sanitize_removed": stats.chars_removed,
        "suspicious": stats.suspicious,
        "pipeline_version": PIPELINE_VERSION,
    }
    # EMBEDDED + INDEXED（add_document 内部完成向量化与入库）
    ok = store.add_document(parents, children, info)
    if not ok:
        return IngestResult(doc_id=did, status="skipped", elapsed_s=time.time() - t0)
    store.save()  # 逐文档落盘 → 可断点续跑

    return IngestResult(
        doc_id=did,
        status="indexed",
        n_parents=len(parents),
        n_children=len(children),
        chars_removed=stats.chars_removed,
        suspicious=stats.suspicious,
        elapsed_s=round(time.time() - t0, 1),
    )


def ingest_paths(
    paths: list[str | Path],
    *,
    store: VectorStore | None = None,
    cfg: RetrievalConfig | None = None,
    sanitize: bool = True,
    limit: int = 0,
) -> list[IngestResult]:
    """批量摄取（幂等）：内容未变的已索引文档直接跳过。limit>0 时只取前 N 个文件（试跑用）。"""
    store = store or VectorStore()
    cfg = cfg or load_retrieval_config()
    files = discover_files(paths)
    if limit > 0:
        files = files[:limit]
    logger.info("[ingest] 发现 %d 个文件", len(files))
    results: list[IngestResult] = []
    for i, fp in enumerate(files, start=1):
        did = doc_id_of(fp)
        chash = compute_content_hash(fp)
        if store.has_document(did, chash):
            logger.info("[ingest] [%d/%d] %s —— 已索引且内容未变，跳过", i, len(files), did)
            results.append(IngestResult(doc_id=did, status="skipped"))
            continue
        try:
            r = ingest_file(fp, store, cfg, sanitize=sanitize)
            results.append(r)
            logger.info("[ingest] [%d/%d] %s —— %s（父 %d / 子 %d，%.1fs）",
                        i, len(files), did, r.status, r.n_parents, r.n_children, r.elapsed_s)
        except Exception as e:  # noqa: BLE001 —— 单文档失败不拖垮批处理
            logger.error("[ingest] [%d/%d] %s —— 失败: %s", i, len(files), did, e)
            results.append(IngestResult(doc_id=did, status="failed", error=str(e)))
    return results


def verify_store(store: VectorStore) -> dict:
    """VERIFIED：索引自检（行数 / 维度 / 文档数一致性）。"""
    dim = store.index.d if store.index is not None else None
    meta_dim = (store.meta or {}).get("dim")
    indexed_docs = [v for v in store.manifest.values() if v.get("status") == "indexed"]
    ok = (
        store.index is not None
        and store.index.ntotal == len(store.child_documents)
        and (meta_dim is None or meta_dim == dim)
    )
    return {
        "ok": ok,
        "n_chunks": store.count,
        "dim": dim,
        "meta_dim": meta_dim,
        "n_docs": len(indexed_docs),
        "embed_model": (store.meta or {}).get("embed_model"),
    }
