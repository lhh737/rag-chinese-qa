"""运行一次评测（薄封装 EvaluationRunner，供 P3 基线 / P5 消融 / P6 防御对照复用）。

示例：
    uv run python scripts/run_eval.py --suite quality --split dev --mode bare \
        --notes "P3 基线裸跑（dev）"
    uv run python scripts/run_eval.py --suite safety --mode bare --store faiss_db \
        --include-types injection_query,harmful,boundary
    uv run python scripts/run_eval.py --suite safety --mode bare --store faiss_db_safety \
        --include-types injection_doc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ragqa.core import RAGPipeline  # noqa: E402
from ragqa.evaluation.runner import EvaluationRunner, RunConfig  # noqa: E402
from ragqa.retrieval.store import VectorStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/golden/golden_v1.jsonl")
    parser.add_argument("--suite", default="quality", choices=["quality", "safety"])
    parser.add_argument("--split", default=None, help="dev | holdout | all（默认全部）")
    parser.add_argument("--mode", default="bare", choices=["bare", "judged"])
    parser.add_argument("--store", default="faiss_db", help="索引目录（注入题用 faiss_db_safety）")
    parser.add_argument("--include-types", default=None, help="逗号分隔：仅跑这些题型/safety_type")
    parser.add_argument("--max-cost", type=float, default=5.0)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--run-type", default="baseline")
    parser.add_argument("--change-type", default="baseline")
    parser.add_argument("--change-description", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    dataset = args.dataset
    if dataset == "data/golden/golden_v1.jsonl" and args.suite == "safety":
        dataset = "data/golden/safety_v1.jsonl"
    store = VectorStore(persist_dir=REPO_ROOT / args.store)
    pipeline = RAGPipeline(store=store)
    cfg = RunConfig(
        dataset_path=REPO_ROOT / dataset,
        suite=args.suite,
        mode=args.mode,
        split=args.split,
        concurrency=args.concurrency,
        max_cost=args.max_cost,
        run_type=args.run_type,
        change_type=args.change_type,
        change_description=args.change_description,
        notes=args.notes,
        run_id=args.run_id,
        include_types=[t for t in args.include_types.split(",") if t] if args.include_types else None,
    )
    summary = EvaluationRunner(pipeline, cfg).run()
    print(f"\nrun_id={summary['run_id']} | {summary['n_rows']}/{summary['n_items']} 题 | "
          f"成本 ¥{summary['cost']['total_cny']:.4f} | 墙钟 {summary['latency']['wall_min']} 分钟")
    print(f"报告: {summary['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
