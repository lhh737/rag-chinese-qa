"""依赖迁移对照实验 · new phase —— 新栈导出 + 与旧栈对照。

运行（从仓库根目录，新环境 .venv）：
    uv run python scripts/parity_check_new.py            # 导出 new_stack.json
    uv run python scripts/parity_check_new.py --compare  # 与 old_stack.json 对照，产出 compare 报告

对照口径（与 parity_check_old.py 完全一致；sanitize 关闭，隔离"依赖迁移"单一变更）：
- 每文档父/子块边界（内容 sha1 列表）逐位置比对
- 5 条查询：向量 top-k / BM25 top-k / 最终父块的 sha1 顺序比对
结果进 decisions/DR-001-deps.md。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

# 隔离目录：不污染正式 faiss_db（与 old phase 相同做法）
PARITY_TMP = os.path.join(REPO_ROOT, ".parity")
os.environ["FAISS_PERSIST_DIR"] = os.path.join(PARITY_TMP, "new_index")

from ragqa.config.loader import RetrievalConfig, load_retrieval_config  # noqa: E402
from ragqa.ingestion.chunker import build_parent_child  # noqa: E402
from ragqa.ingestion.loader import compute_content_hash, load_file  # noqa: E402
from ragqa.retrieval.hybrid import HybridRetriever  # noqa: E402
from ragqa.retrieval.store import VectorStore  # noqa: E402

# 与 old phase 保持一致的固定输入
DOC_FILES = [
    "data/papers/1706.03762.pdf",
    "data/papers/2005.11401.pdf",
    "data/papers/2005.14165.pdf",
    "data/papers/2204.05862.pdf",
    "data/papers/2205.01068.pdf",
    "data/papers/2302.13971.pdf",
    "data/papers/2303.08774.pdf",
    "data/papers/2312.10997.pdf",
    "data/llm_intro.txt",
    "data/test_transformer.txt",
]

QUERIES = [
    "Transformer 的注意力机制计算复杂度是多少？",
    "RAG 是什么？它解决了什么问题？",
    "GPT-3 模型有多少参数？",
    "LLaMA 模型的训练数据规模是多少？",
    "什么是大语言模型？",
]

# 对照必须使用旧栈的行为：merge 融合（RRF 是后续的独立变更）
PARITY_CFG = RetrievalConfig(fusion="merge", hyde=load_retrieval_config().hyde)


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def head(text: str, n: int = 60) -> str:
    return text[:n].replace("\n", " ")


def doc_id(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def export() -> None:
    from ragqa.utils.paths import resolve_path

    store = VectorStore(persist_dir=os.path.join(PARITY_TMP, "new_index"))
    out: dict = {
        "phase": "new",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {"fusion": "merge", "note": "sanitize=off；hyde=off（与 old phase 相同口径）"},
        "documents": {},
        "queries": [],
    }

    print("[parity-new] loading documents + building temp index ...")
    for rel in DOC_FILES:
        path = resolve_path(rel)
        did = doc_id(str(path))
        t0 = time.time()
        pages = load_file(path)
        parents, children = build_parent_child(pages, PARITY_CFG)  # 不经过 sanitize
        info = {
            "doc_id": did,
            "content_hash": compute_content_hash(path),
            "path": str(path),
            "filename": path.name,
            "status": "indexed",
            "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        store.add_document(parents, children, info)
        store.save()

        pid_to_idx = {p.metadata["parent_id"]: i for i, p in enumerate(parents)}
        out["documents"][did] = {
            "file": rel,
            "n_parents": len(parents),
            "n_children": len(children),
            "parents": [{"i": i, "sha1": sha1(p.page_content), "len": len(p.page_content), "head": head(p.page_content)} for i, p in enumerate(parents)],
            "children": [
                {"i": i, "sha1": sha1(c.page_content), "len": len(c.page_content), "head": head(c.page_content),
                 "parent_idx": pid_to_idx.get(c.metadata.get("parent_id"), -1)}
                for i, c in enumerate(children)
            ],
        }
        print(f"[parity-new]   {did}: parents={len(parents)} children={len(children)} ({time.time() - t0:.1f}s)")

    hybrid = HybridRetriever(store, PARITY_CFG)
    for q in QUERIES:
        t0 = time.time()
        vec = store.search_with_scores(q, PARITY_CFG.vector_fetch_k)
        bm = store.bm25_rank(q, PARITY_CFG.bm25_fetch_k)
        result = hybrid.retrieve(q)
        out["queries"].append({
            "q": q,
            "elapsed_s": round(time.time() - t0, 1),
            "vector_top": [{"sha1": sha1(d.page_content), "dist": round(float(s), 6), "len": len(d.page_content)} for d, s in vec],
            "bm25_top": [{"sha1": sha1(d.page_content), "len": len(d.page_content)} for d, _s in bm],
            "final_parents": [{"sha1": sha1(b.text), "source": b.source, "len": len(b.text)} for b in result.blocks],
            "context_chars": result.trace.context_chars,
        })
        print(f"[parity-new]   Q: {q}  final_parents={len(result.blocks)} ({out['queries'][-1]['elapsed_s']}s)")

    path = os.path.join(REPO_ROOT, "eval_results", "parity", "new_stack.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[parity-new] saved -> {path}")


def compare() -> int:
    base = os.path.join(REPO_ROOT, "eval_results", "parity")
    old = json.loads(open(os.path.join(base, "old_stack.json"), encoding="utf-8").read())
    new = json.loads(open(os.path.join(base, "new_stack.json"), encoding="utf-8").read())

    report: dict = {"docs": {}, "queries": [], "summary": {}}
    total_pos = total_match = 0

    for did in old["documents"]:
        o, n = old["documents"][did], new["documents"].get(did, {})
        row = {"n_parents": (o["n_parents"], n.get("n_parents")), "n_children": (o["n_children"], n.get("n_children"))}
        for part in ("parents", "children"):
            os_ = [c["sha1"] for c in o[part]]
            ns = [c["sha1"] for c in n.get(part, [])]
            matched = sum(1 for a, b in zip(os_, ns) if a == b)
            total_pos += max(len(os_), len(ns))
            total_match += matched
            row[part] = {"old_n": len(os_), "new_n": len(ns), "matched": matched,
                         "rate": round(matched / max(len(os_), len(ns), 1), 4)}
        report["docs"][did] = row

    for oq, nq in zip(old["queries"], new["queries"]):
        def list_match(key: str) -> dict:
            os_ = [x["sha1"] for x in oq[key]]
            ns = [x["sha1"] for x in nq[key]]
            return {"old_n": len(os_), "new_n": len(ns),
                    "order_match": os_ == ns,
                    "set_overlap": len(set(os_) & set(ns)) / max(len(set(os_) | set(ns)), 1)}
        row = {"q": oq["q"], "vector": list_match("vector_top"), "bm25": list_match("bm25_top"), "final": list_match("final_parents")}
        report["queries"].append(row)

    report["summary"] = {
        "chunk_boundary_match_rate": round(total_match / max(total_pos, 1), 4),
        "n_chunk_positions": total_pos,
        "all_query_final_identical": all(r["final"]["order_match"] for r in report["queries"]),
        "all_query_vector_order_identical": all(r["vector"]["order_match"] for r in report["queries"]),
        "all_query_bm25_order_identical": all(r["bm25"]["order_match"] for r in report["queries"]),
    }

    out_path = os.path.join(base, "compare_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    print("=== 依赖迁移对照结果 ===")
    print(f"chunk 边界一致率: {report['summary']['chunk_boundary_match_rate'] * 100:.3f}% ({total_pos} 个位置)")
    for r in report["queries"]:
        print(f"Q: {r['q'][:24]}… vector序同={r['vector']['order_match']} bm25序同={r['bm25']['order_match']} 最终父块序同={r['final']['order_match']} (最终 {r['final']['old_n']}块)")
    print(f"报告 -> {out_path}")
    ok = report["summary"]["chunk_boundary_match_rate"] >= 0.999 and report["summary"]["all_query_final_identical"]
    print("[parity] 通过标准（chunk≥99.9% 且 5 查询最终一致）:", "PASS ✓" if ok else "FAIL ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="与 old_stack.json 对照")
    args = parser.parse_args()
    if args.compare:
        sys.exit(compare())
    export()
