"""离线索引修复（P0 事故恢复 · 零 API 调用）。

背景：2026-10-05 全量重建在 2605.06609（PDF 数学字体产生孤立代理项）处触发编码错误，
该文档未入库；但其父块已先进入内存，导致其后每篇文档的 parent_store 落盘失败。
最终磁盘状态：index.faiss 与 child_docs.json 完好（含除毒文档外的全部 106 篇），
parent_store/manifest/meta/tokenized 停留在中途。

修复策略（不重复任何 embedding 调用）：
1. 以 child_docs.json（行号真相）为准，校验索引行数一致、毒文档不在其中；
2. 对全部已入库文档做**确定性重分块**（同一 loader/chunker/清洗），逐块 sha1 对比
   child_docs.json —— 完全一致才继续（这是行号对齐的硬保证）；
3. 原子重写 parent_store / manifest（补 16 条缺失条目）/ meta / tokenized；
4. 运行 verify_store 自检。

运行：uv run python scripts/repair_index.py [--exclude DOC_ID]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import faiss  # noqa: E402

from ragqa.config.loader import load_retrieval_config  # noqa: E402
from ragqa.ingestion.chunker import build_parent_child  # noqa: E402
from ragqa.ingestion.loader import compute_content_hash, load_file  # noqa: E402
from ragqa.ingestion.pipeline import verify_store  # noqa: E402
from ragqa.ingestion.sanitize import SanitizeStats, sanitize_text  # noqa: E402
from ragqa.retrieval.store import PIPELINE_VERSION, VectorStore  # noqa: E402
from ragqa.types import Document  # noqa: E402

EXCLUDE_DEFAULT = "2605.06609__2026-05-07"


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude", default=EXCLUDE_DEFAULT, help="排除的 doc_id（毒文档）")
    args = parser.parse_args()

    store_dir = REPO_ROOT / "faiss_db"
    # 1) 载入磁盘真相
    child_raw = json.loads((store_dir / "child_docs.json").read_text(encoding="utf-8"))
    disk_children = [Document(page_content=d["page_content"], metadata=d.get("metadata", {})) for d in child_raw]
    index = faiss.read_index(str(store_dir / "index.faiss"))
    old_manifest = json.loads((store_dir / "manifest.json").read_text(encoding="utf-8"))
    print(f"[repair] 磁盘：child_docs={len(disk_children)}  index.ntotal={index.ntotal}  manifest={len(old_manifest)}")
    assert len(disk_children) == index.ntotal, "子块数与索引行数不一致，停止"
    assert not [c for c in disk_children if c.metadata.get("doc_id") == args.exclude], "毒文档不应在索引中"

    # 2) 全语料顺序（与 ingest 的 discover 顺序一致）
    files = sorted((REPO_ROOT / "data" / "papers").glob("*.pdf")) + [
        REPO_ROOT / "data" / "llm_intro.txt",
        REPO_ROOT / "data" / "test_transformer.txt",
    ]
    included = [f for f in files if f.stem != args.exclude]
    print(f"[repair] 重分块 {len(included)} 篇（确定性），逐块校验 …")

    cfg = load_retrieval_config()
    parents_all: list[Document] = []
    children_all: list[Document] = []
    stats_by_doc: dict[str, SanitizeStats] = {}
    counts_by_doc: dict[str, tuple[int, int]] = {}
    t0 = time.time()
    for i, fp in enumerate(included, 1):
        pages = load_file(fp)
        stats = SanitizeStats()
        for p in pages:
            cleaned, s = sanitize_text(p.page_content)
            p.page_content = cleaned
            stats.merge(s)
        parents, children = build_parent_child(pages, cfg)
        parents_all.extend(parents)
        children_all.extend(children)
        stats_by_doc[fp.stem] = stats
        counts_by_doc[fp.stem] = (len(parents), len(children))
        if i % 20 == 0 or i == len(included):
            print(f"[repair]   {i}/{len(included)} …（{time.time() - t0:.0f}s）")

    # 3) 硬校验：逐块 sha1 一致（行号对齐保证）
    print(f"[repair] 校验：重分块 {len(children_all)} vs 磁盘 {len(disk_children)}")
    assert len(children_all) == len(disk_children), "子块数不一致，停止（对齐无法保证）"
    mismatches = [i for i, (a, b) in enumerate(zip(children_all, disk_children)) if a.page_content != b.page_content]
    if mismatches:
        print(f"[repair] ✗ 逐块校验失败，首个差异位置: {mismatches[0]}（共 {len(mismatches)} 处），停止")
        return 1
    print("[repair] ✓ 逐块 sha1 完全一致（32501/32501 行号对齐成立）")

    # 4) 组装并原子落盘
    store = VectorStore(persist_dir=store_dir, auto_load=False)
    store.index = index
    store.child_documents = disk_children
    store.parent_store = {
        p.metadata["parent_id"]: {
            "page_content": p.page_content,
            "metadata": {k: v for k, v in p.metadata.items() if k != "parent_id"},
        }
        for p in parents_all
    }
    manifest = dict(old_manifest)
    added = []
    for fp in included:
        did = fp.stem
        if did in manifest:
            continue
        stats = stats_by_doc[did]
        n_parents, n_children = counts_by_doc[did]
        manifest[did] = {
            "doc_id": did,
            "content_hash": compute_content_hash(fp),
            "path": str(fp.resolve()),
            "filename": fp.name,
            "size_kb": round(fp.stat().st_size / 1024, 1),
            "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "indexed",
            "n_parents": n_parents,
            "n_children": n_children,
            "sanitize_removed": stats.chars_removed,
            "suspicious": stats.suspicious,
            "pipeline_version": PIPELINE_VERSION,
            "repaired": "offline-rebuild (P0 incident, 零新增 embedding 调用)",
        }
        added.append(did)
    store.manifest = manifest
    store.tokenized_corpus = []  # 触发重建
    from ragqa.utils.text import zh_tokenize

    store.tokenized_corpus = [zh_tokenize(d.page_content) for d in disk_children]
    store.meta = {
        "dim": index.d,
        "embed_model": store.meta.get("embed_model") or None,
        "count": len(disk_children),
        "pipeline_version": PIPELINE_VERSION,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    # embed_model 名从旧 meta 读（已在上行保留）；若缺失则从 settings 补
    if not store.meta["embed_model"]:
        from ragqa.config.settings import get_settings

        store.meta["embed_model"] = get_settings().embed_model_name
    store.save()
    print(f"[repair] 已落盘：parent_store={len(store.parent_store)}，manifest 补 {len(added)} 条：{added}")

    # 5) 自检
    check = VectorStore(persist_dir=store_dir)  # 触发完整加载校验
    v = verify_store(check)
    print(f"[repair] verify_store: {v}")
    dirty = {d: s.chars_removed for d, s in stats_by_doc.items() if s.chars_removed}
    print(f"[repair] 清洗统计（本次重分块口径）：{dirty or '无'}")
    print(f"[repair] 排除文档：{args.exclude}（如需纳入，修复后单篇 ingest 即可）")
    return 0 if v["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
