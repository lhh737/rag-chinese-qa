"""对某 run 中 judge=error 的条目复判并回填（瞬态错误治理，P5 复用）。

背景：裁判偶发空输出/截断（"无 JSON"）会产生 judge_error——设计口径为"不计入分母、
如实申报"，但通常复跑一次即可补齐。本工具就地复判错误条目、回填 results.jsonl 并重建报告。

用法：
    uv run python scripts/rejudge_errors.py --run-id 20261013-003
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from ragqa.config.settings import get_settings  # noqa: E402
from ragqa.evaluation.datasets import load_jsonl  # noqa: E402
from ragqa.evaluation.judges.cache import JudgeCache  # noqa: E402
from ragqa.evaluation.judges.client import JudgeClient  # noqa: E402
from ragqa.evaluation.report.builder import build_report  # noqa: E402
from ragqa.evaluation.runner import EvaluationRunner, RunConfig  # noqa: E402
from ragqa.models.factory import get_judge_client  # noqa: E402
from scripts.judge_crosscheck import build_payload  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--prompt-version", default="v1")
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    run_dir = REPO_ROOT / "eval_results" / args.run_id
    results_path = run_dir / "results.jsonl"
    rows = [json.loads(l) for l in results_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    traces = {json.loads(l)["trace_id"]: json.loads(l) for l in
              (run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    items = {i.id: i for i in load_jsonl(REPO_ROOT / "data/golden/golden_v1.jsonl")}
    items.update({i.id: i for i in load_jsonl(REPO_ROOT / "data/golden/safety_v1.jsonl")})

    errs = [(r, cr) for r in rows for cr, v in (r.get("judge") or {}).items()
            if (v or {}).get("verdict") == "error"]
    if not errs:
        print(f"{args.run_id}: 无 judge error，无需复判")
        return 0
    print(f"{args.run_id}: 发现 {len(errs)} 条 judge error，开始复判…")

    s = get_settings()
    judge = JudgeClient(model=s.judge_model, cache=JudgeCache(), client=get_judge_client(),
                        extra_body={}, max_tokens=args.max_tokens)
    patched = 0
    for row, cr in errs:
        item = items.get(row["id"])
        trace = traces.get(row.get("trace_id", "")) or {}
        answer = (trace.get("generation") or {}).get("answer") or ""
        try:
            v = judge.judge(item_id=row["id"], answer=answer, criterion=cr,
                            prompt_version=args.prompt_version, payload=build_payload(item, answer, trace))
            row["judge"][cr] = v
            patched += 1
            print(f"  ✓ {row['id']}/{cr} → {v.get('verdict')}")
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {row['id']}/{cr} 仍失败: {str(e)[:120]}")

    if patched:
        bak = run_dir / "results.jsonl.bak_pre_rejudge"
        if not bak.exists():
            shutil.copy2(results_path, bak)
        tmp = results_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(results_path)

        run_items = [items[r["id"]] for r in rows if r["id"] in items]
        cfg = RunConfig(dataset_path=REPO_ROOT / "data/golden/golden_v1.jsonl",
                        suite=rows[0].get("suite", "quality"), mode="judged")
        summary = EvaluationRunner(None, cfg)._summarize(rows, run_items, args.run_id, aborted=False, wall_s=0.0)
        summary["out_dir"] = str(run_dir)
        report_path = build_report(summary, run_dir)
        cov = summary.get("coverage") or {}
        print(f"回填 {patched}/{len(errs)} | 覆盖率 {cov.get('judge_coverage')}（{cov.get('judge_valid')}/{cov.get('judge_expected')}）")
        print(f"报告已重建：{report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
