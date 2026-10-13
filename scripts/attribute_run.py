"""对某个 judged run 跑自动归因（taxonomy v1）+ 生成人工抽检文件（查看器模式③）。

用法：
    uv run python scripts/attribute_run.py --run-id 20261013-003 --spot-check 20

输出：
- eval_results/<run_id>/attribution.jsonl   （逐题归因 + 证据）
- eval_results/<run_id>/spotcheck_<run_id>.jsonl（人工抽检样本：失败项为主 + 少量通过项作对照）
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ragqa.evaluation.attribution import attribute_item, suggestion_of  # noqa: E402
from ragqa.evaluation.datasets import load_jsonl  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--spot-check", type=int, default=20)
    args = ap.parse_args()

    run_dir = REPO_ROOT / "eval_results" / args.run_id
    rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    traces = {json.loads(l)["trace_id"]: json.loads(l) for l in
              (run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    items = {i.id: i for i in load_jsonl(REPO_ROOT / "data/golden/golden_v1.jsonl")}
    items.update({i.id: i for i in load_jsonl(REPO_ROOT / "data/golden/safety_v1.jsonl")})

    out = []
    for r in rows:
        item = items.get(r["id"])
        trace = traces.get(r.get("trace_id", "")) or {}
        if item is None:
            continue
        a = attribute_item(item, r, trace)
        out.append({"id": r["id"], "type": item.type or item.safety_type,
                    "question": item.question,
                    "answer": (trace.get("generation") or {}).get("answer") or "",
                    "attribution": a["primary"], "secondary": a.get("secondary", []),
                    "evidence": a.get("evidence", {}),
                    "suggestion": suggestion_of(a["primary"]),
                    "verdicts": {cr: v.get("verdict") for cr, v in (r.get("judge") or {}).items()}})

    attr_path = run_dir / "attribution.jsonl"
    with attr_path.open("w", encoding="utf-8") as f:
        for a in out:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    counts = Counter(a["attribution"] for a in out)
    print(f"归因完成 {len(out)} 题：")
    for k, v in counts.most_common():
        print(f"  {k}: {v}")

    # 抽检样本：失败项为主 + 少量通过项对照（按题号取样，确定性）
    failed = [a for a in out if a["attribution"] != "无问题"]
    passed = [a for a in out if a["attribution"] == "无问题"]
    n_fail_take = max(0, args.spot_check - min(3, len(passed)))
    sample = failed[:n_fail_take] + passed[: min(3, len(passed))]
    spot = [{
        "case_id": f'{args.run_id}:{a["id"]}', "question": a["question"], "answer": a["answer"],
        "attribution": a["attribution"],
        "sub_label": "; ".join(a["secondary"]) if a["secondary"] else "",
        "evidence": json.dumps(a["evidence"], ensure_ascii=False)[:600],
        "suggestion": a["suggestion"],
    } for a in sample]
    spot_path = run_dir / f"spotcheck_{args.run_id}.jsonl"
    with spot_path.open("w", encoding="utf-8") as f:
        for s in spot:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"归因: {attr_path}\n抽检样本: {spot_path}（{len(spot)} 条：失败 {n_fail_take} + 通过对照 {min(3, len(passed))}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
