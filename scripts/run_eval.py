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

from ragqa.config.loader import load_retrieval_config  # noqa: E402
from ragqa.config.settings import get_settings  # noqa: E402
from ragqa.core import RAGPipeline  # noqa: E402
from ragqa.evaluation.judges.client import JudgeClient  # noqa: E402
from ragqa.evaluation.runner import EvaluationRunner, RunConfig  # noqa: E402
from ragqa.models.factory import get_judge_client  # noqa: E402
from ragqa.retrieval.store import VectorStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/golden/golden_v1.jsonl")
    parser.add_argument("--suite", default="quality", choices=["quality", "safety"])
    parser.add_argument("--split", default=None, help="dev | holdout | all（默认全部）")
    parser.add_argument("--mode", default="bare", choices=["bare", "judged"])
    parser.add_argument("--store", default="faiss_db", help="索引目录（注入题用 faiss_db_safety）")
    parser.add_argument("--include-types", default=None, help="逗号分隔：仅跑这些题型/safety_type")
    parser.add_argument("--limit", type=int, default=None, help="仅取前 N 题（冒烟用）")
    parser.add_argument("--no-rerank", action="store_true", help="消融：移除重排（回退融合序）")
    parser.add_argument("--no-bm25", action="store_true", help="消融：纯稠密检索")
    parser.add_argument("--no-hyde", action="store_true", help="消融：关闭 HyDE")
    parser.add_argument("--fusion", default=None, choices=["rrf", "merge"], help="融合策略替换（消融 A4）")
    parser.add_argument("--rrf-k", type=int, default=None, help="RRF k 值（扫描用）")
    parser.add_argument("--base-context-k", type=int, default=None, help="最终上下文基数（迭代实验）")
    parser.add_argument("--max-context-k", type=int, default=None, help="最终上下文上限（迭代实验）")
    parser.add_argument("--gen-prompt", default=None, help="生成 prompt 版本（如 v2；缺省按 config）")
    parser.add_argument("--pipeline", default="fixed", choices=["fixed", "agentic"],
                        help="fixed=现有固定管线；agentic=单轮工具选择变体（P9-D，需配 --run-type agentic）")
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
    rcfg = load_retrieval_config()
    updates = {}
    if args.no_rerank:
        updates["rerank_enabled"] = False
    if args.no_bm25:
        updates["bm25_enabled"] = False
    if args.fusion:
        updates["fusion"] = args.fusion
    if args.rrf_k is not None:
        updates["rrf_k"] = args.rrf_k
    if args.base_context_k is not None:
        updates["base_context_k"] = args.base_context_k
    if args.max_context_k is not None:
        updates["max_context_k"] = args.max_context_k
    if updates:
        rcfg = rcfg.model_copy(update=updates)
    if args.no_hyde:
        rcfg = rcfg.model_copy(update={"hyde": rcfg.hyde.model_copy(update={"enabled": False})})
    gcfg = None
    if args.gen_prompt:
        from ragqa.config.loader import load_generation_config
        gcfg = load_generation_config().model_copy(update={"prompt_version": args.gen_prompt})
    pipeline = RAGPipeline(store=store, retrieval_config=rcfg, generation_config=gcfg)
    if args.pipeline == "agentic":
        from ragqa.agentic import ToolCallingPipeline

        pipeline = ToolCallingPipeline(pipeline)
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
        limit=args.limit,
    )
    judge_client = None
    if args.mode == "judged":
        s = get_settings()
        if not s.judge_model:
            print("✗ judged 模式需要裁判模型（.env JUDGE_MODEL）")
            return 2
        # 与 P4 校准完全一致的裁判配置：extra_body={}（不关思考）、max_tokens 2048
        judge_client = JudgeClient(model=s.judge_model, client=get_judge_client(),
                                   extra_body={}, max_tokens=2048)
        print(f"裁判: {s.judge_model} @ {s.judge_base_url or '(默认端点)'}")
    summary = EvaluationRunner(pipeline, cfg, judge_client=judge_client).run()
    print(f"\nrun_id={summary['run_id']} | {summary['n_rows']}/{summary['n_items']} 题 | "
          f"成本 ¥{summary['cost']['total_cny']:.4f} | 墙钟 {summary['latency']['wall_min']} 分钟")
    print(f"报告: {summary['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
