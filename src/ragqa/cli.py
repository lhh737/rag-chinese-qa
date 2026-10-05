"""ragqa CLI（console script：`uv run ragqa <子命令>`）。

  ragqa ingest [路径...] [--no-sanitize] [--limit N]   # 数据管道导入（幂等，可断点续跑）
  ragqa documents                                      # 已索引文档清单
  ragqa query "问题" [--prompt-version vN] [--trace-out FILE] [--json]
  ragqa smoke [--n-docs 20] [--out DIR]                # 冒烟：小索引 + 固定查询 + 小评测（Recall@5）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ragqa.utils.paths import REPO_ROOT

# 默认导入范围：105 篇论文 + 2 个有内容的中文文档（36 字节的特殊字符文件名测试文件排除）
DEFAULT_INGEST_PATHS = ["data/papers", "data/llm_intro.txt", "data/test_transformer.txt"]

# 冒烟固定查询（含期望支撑文档，用于 Recall@5 小评测）
SMOKE_QUERIES = [
    {"q": "Transformer 的注意力机制计算复杂度是多少？", "expected": ["1706.03762"]},
    {"q": "RAG 是什么？它解决了什么问题？", "expected": ["2005.11401"]},
    {"q": "GPT-3 模型有多少参数？", "expected": ["2005.14165"]},
    {"q": "LLaMA 模型的训练数据规模是多少？", "expected": ["2302.13971"]},
    {"q": "什么是大语言模型？", "expected": ["llm_intro"]},
]


# ── ingest ─────────────────────────────────────────────────

def cmd_ingest(args: argparse.Namespace) -> int:
    from ragqa.config.loader import load_pricing_config
    from ragqa.ingestion.pipeline import ingest_paths, verify_store
    from ragqa.retrieval.store import VectorStore

    store = VectorStore()
    paths = args.paths or DEFAULT_INGEST_PATHS
    t0 = time.time()
    results = ingest_paths(paths, store=store, sanitize=not args.no_sanitize, limit=args.limit)

    n_indexed = sum(1 for r in results if r.status == "indexed")
    n_skipped = sum(1 for r in results if r.status == "skipped")
    n_failed = sum(1 for r in results if r.status == "failed")
    v = verify_store(store)

    tokens = int(getattr(store.embed, "total_tokens", 0) or 0)
    rate = (load_pricing_config().embed.get(store.meta.get("embed_model", ""), {}) or {}).get("input", 0.0)
    cost = tokens * rate / 1_000_000

    print(f"\n导入完成：新增 {n_indexed} / 跳过 {n_skipped} / 失败 {n_failed}（总耗时 {time.time() - t0:.0f}s）")
    print(f"索引自检：{v}")
    if tokens:
        print(f"本次 embedding tokens={tokens}，估算成本 ≈ {cost:.4f} 元")
    for r in results:
        if r.status == "failed":
            print(f"  ✗ {r.doc_id}: {r.error}")
    return 1 if n_failed else 0


# ── documents ──────────────────────────────────────────────

def cmd_documents(args: argparse.Namespace) -> int:
    from ragqa.retrieval.store import VectorStore

    store = VectorStore()
    files = store.file_list
    if not files:
        print("知识库为空（faiss_db 不存在或未导入），先运行: ragqa ingest")
        return 0
    print(f"共 {len(files)} 篇文档，{store.count} 个子块：\n")
    for f in files:
        print(f"  {f.get('doc_id', '?'):<24} {f.get('n_children', '?'):>5} 块  {f.get('loaded_at', '')}  {f.get('filename', '')}")
    return 0


# ── query ──────────────────────────────────────────────────

def cmd_query(args: argparse.Namespace) -> int:
    from ragqa.core import RAGPipeline
    from ragqa.trace import TraceWriter

    pipe = RAGPipeline()
    t0 = time.time()
    res = pipe.query(args.question, prompt_version=args.prompt_version)
    elapsed = time.time() - t0

    if args.json:
        print(json.dumps(
            {"answer": res.answer, "citations": [c.__dict__ for c in res.citations], "trace": res.trace},
            ensure_ascii=False, indent=1,
        ))
    else:
        print(f"\n【答】{res.answer}\n")
        if res.contexts:
            print("【参考】")
            for b in res.contexts:
                print(f"  [{b.rank}] {b.source} ({b.doc_id}) score={b.rerank_score}")
        cs = res.trace.get("totals", {})
        print(f"\n【耗时 {elapsed:.1f}s / 估算成本 {cs.get('cost_total_cny', 0):.4f} 元】")

    if args.trace_out:
        TraceWriter(args.trace_out).append(res.trace)
        print(f"trace 已写入 {args.trace_out}")
    return 0


# ── smoke ──────────────────────────────────────────────────

def cmd_smoke(args: argparse.Namespace) -> int:
    from ragqa.config.loader import load_pricing_config
    from ragqa.core import RAGPipeline
    from ragqa.ingestion.pipeline import ingest_paths, verify_store
    from ragqa.retrieval.store import VectorStore
    from ragqa.trace import TraceWriter

    out_dir = Path(args.out) if args.out else REPO_ROOT / "eval_results" / f"smoke-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 小规模重建索引（隔离目录，不污染正式 faiss_db）
    store = VectorStore(persist_dir=out_dir / "index")
    papers = sorted((REPO_ROOT / "data" / "papers").glob("*.pdf"))[: args.n_docs]
    paths: list[str] = [str(p) for p in papers] + [
        str(REPO_ROOT / "data" / "llm_intro.txt"),
        str(REPO_ROOT / "data" / "test_transformer.txt"),
    ]
    print(f"[smoke] 导入 {len(paths)} 个文档 → {out_dir / 'index'}")
    t0 = time.time()
    ingest_results = ingest_paths(paths, store=store)
    ingest_s = time.time() - t0
    verify = verify_store(store)
    if not verify["ok"]:
        print(f"[smoke] ✗ 索引自检失败: {verify}")
        return 1

    # 2) 固定查询 + trace 落盘 + Recall@5（文档级，取融合后 top5 的文档）
    pipe = RAGPipeline(store=store)
    writer = TraceWriter(out_dir / "traces.jsonl")
    rows = []
    for item in SMOKE_QUERIES[: args.n_queries]:
        res = pipe.query(item["q"])
        writer.append(res.trace)
        top5_docs: list[str] = []
        for f in ((res.trace.get("rrf") or {}).get("fused") or []):
            doc = f["chunk_uid"].split(":")[0]
            if doc not in top5_docs:
                top5_docs.append(doc)
            if len(top5_docs) >= 5:
                break
        recall = len(set(item["expected"]) & set(top5_docs)) / len(item["expected"])
        rows.append({
            "q": item["q"], "expected": item["expected"], "top5_docs": top5_docs,
            "recall@5": recall,
            "answer": res.answer,
            "citation_stats": res.trace.get("generation", {}).get("citation_stats", {}) if res.trace.get("generation") else {},
            "latency_ms": res.trace.get("totals", {}).get("latency_ms", {}).get("total", 0),
            "cost_cny": res.trace.get("totals", {}).get("cost_total_cny", 0),
        })
        print(f"[smoke] Q: {item['q']}  recall@5={recall:.2f}  ({rows[-1]['latency_ms']:.0f}ms)")

    avg_recall = sum(r["recall@5"] for r in rows) / len(rows) if rows else 0.0
    total_cost = sum(r["cost_cny"] for r in rows)
    embed_tokens = int(getattr(store.embed, "total_tokens", 0) or 0)
    embed_rate = (load_pricing_config().embed.get(store.meta.get("embed_model", ""), {}) or {}).get("input", 0.0)
    embed_cost = embed_tokens * embed_rate / 1_000_000

    # 3) 报告
    lines = [
        "# P0 冒烟报告（smoke）",
        "",
        f"- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 索引：{verify['n_docs']} 文档 / {verify['n_chunks']} 子块 / dim={verify['dim']} / 模型={verify['embed_model']}",
        f"- 导入耗时：{ingest_s:.0f}s（embedding tokens={embed_tokens}，估算成本 ≈ {embed_cost:.4f} 元）",
        f"- 查询数：{len(rows)}；平均 Recall@5（文档级，融合后 top5）：**{avg_recall:.2f}**",
        f"- 查询成本合计（估算）：{total_cost:.4f} 元",
        "",
        "## 逐条结果",
        "",
        "| 查询 | 期望文档 | top5 文档 | Recall@5 | 引用统计 | 延迟ms |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['q']} | {', '.join(r['expected'])} | {', '.join(r['top5_docs'])} "
            f"| {r['recall@5']:.2f} | {r['citation_stats']} | {r['latency_ms']:.0f} |"
        )
    lines += ["", "## 回答与引用（人工抽查用）", ""]
    for r in rows:
        lines += [f"### {r['q']}", "", r["answer"], ""]
    (out_dir / "smoke_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n[smoke] 平均 Recall@5 = {avg_recall:.2f}，报告 → {out_dir / 'smoke_report.md'}")
    return 0


# ── main ───────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragqa", description="RAG 中文问答 · 评测主线 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="数据管道导入（幂等）")
    p_ingest.add_argument("paths", nargs="*", help="文件或目录（默认：data/papers + 2 个中文文档）")
    p_ingest.add_argument("--no-sanitize", action="store_true", help="关闭文档清洗（对照实验用）")
    p_ingest.add_argument("--limit", type=int, default=0, help="最多导入 N 个文件（0=不限）")
    p_ingest.set_defaults(func=cmd_ingest)

    p_docs = sub.add_parser("documents", help="已索引文档清单")
    p_docs.set_defaults(func=cmd_documents)

    p_query = sub.add_parser("query", help="单条查询")
    p_query.add_argument("question")
    p_query.add_argument("--prompt-version", default=None, help="answer prompt 版本（如 v1）")
    p_query.add_argument("--trace-out", default=None, help="trace JSONL 落盘路径")
    p_query.add_argument("--json", action="store_true", help="输出完整 JSON")
    p_query.set_defaults(func=cmd_query)

    p_smoke = sub.add_parser("smoke", help="冒烟：小索引 + 固定查询 + Recall@5 小评测")
    p_smoke.add_argument("--n-docs", type=int, default=20, help="论文数量（默认 20）")
    p_smoke.add_argument("--n-queries", type=int, default=5)
    p_smoke.add_argument("--out", default=None, help="输出目录（默认 eval_results/smoke-<时间>）")
    p_smoke.set_defaults(func=cmd_smoke)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
