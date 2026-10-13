"""P5 移除式消融 · 配对分析面板（设计 §4.11：对照实验 + 配对统计判定）。

对比每个消融 run 与基线 R1：逐题配对（McNemar，连续修正）+ 检索层均值差 +
误拒配对 + multi_hop 专项。输出 markdown 面板。

用法：
    uv run python scripts/ablation_analysis.py --baseline 20261013-003 \
        --run A1-去重排=20261013-004 --run A2-去HyDE=20261013-005 \
        --run A3-纯稠密=20261013-006 --run A4-merge融合=20261013-007
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ragqa.evaluation.metrics import mcnemar  # noqa: E402

CRITERIA = ["answer_correctness", "faithfulness", "citation_accuracy", "oos_refusal"]


def load_run(run_id: str) -> dict[str, dict]:
    path = REPO_ROOT / "eval_results" / run_id / "results.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["id"]: r for r in rows}


def verdict(row: dict, cr: str) -> str | None:
    v = ((row.get("judge") or {}).get(cr) or {}).get("verdict")
    return v if v in ("pass", "fail") else None


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--run", action="append", required=True, help="标签=RUN_ID，可多次")
    args = ap.parse_args()

    base = load_run(args.baseline)
    runs = [(s.split("=", 1)[0], s.split("=", 1)[1]) for s in args.run]

    L = [f"# P5 消融配对面板（基线 {args.baseline}）\n",
         "> 配对口径：逐题对比（同题同评测集）；b=基线过/消融败，c=基线败/消融过；",
         "> p 为 McNemar 连续修正近似（小样本请谨慎解读，见设计 §4.11）。\n"]

    for label, rid in runs:
        ab = load_run(rid)
        ids = sorted(set(base) & set(ab))
        L.append(f"## {label}（`{rid}`）\n")

        # 1) 判定层配对
        L.append("| criterion | 基线过/消融败 (b) | 基线败/消融过 (c) | Δpp | McNemar p | 配对 n |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for cr in CRITERIA:
            pairs = [(verdict(base[q], cr), verdict(ab[q], cr)) for q in ids]
            pairs = [(x, y) for x, y in pairs if x and y]
            if not pairs:
                continue
            n = len(pairs)
            b = sum(1 for x, y in pairs if x == "pass" and y == "fail")
            c = sum(1 for x, y in pairs if x == "fail" and y == "pass")
            mp = mcnemar(b, c)
            pr_base = sum(1 for x, _ in pairs if x == "pass") / n
            pr_ab = sum(1 for _, y in pairs if y == "pass") / n
            delta = (pr_ab - pr_base) * 100
            L.append(f"| {cr} | {b} | {c} | {delta:+.1f} | {mp['p']:.3f} | {n} |")
        L.append("")

        # 2) 误拒配对（规则口径）
        pairs = [(bool(base[q]["metrics"].get("false_refusal")), bool(ab[q]["metrics"].get("false_refusal")))
                 for q in ids if base[q]["metrics"].get("false_refusal") is not None]
        if pairs:
            n = len(pairs)
            b = sum(1 for x, y in pairs if not x and y)   # 基线不拒→消融拒（变差）
            c = sum(1 for x, y in pairs if x and not y)   # 基线拒→消融不拒（变好）
            fr_base = sum(1 for x, _ in pairs if x) / n
            fr_ab = sum(1 for _, y in pairs if y) / n
            mp = mcnemar(b, c)
            L.append(f"**误拒配对**：基线 {fr_base:.3f} → 消融 {fr_ab:.3f}（Δ{(fr_ab - fr_base) * 100:+.1f}pp；"
                     f"新增误拒 b={b} / 减少误拒 c={c}；p={mp['p']:.3f}）\n")

        # 3) 检索层均值差（同一题集合）
        for mname, key in (("Recall@5", "recall@5"), ("MRR", "mrr")):
            vals = [(base[q]["metrics"].get(key), ab[q]["metrics"].get(key)) for q in ids]
            vals = [(x, y) for x, y in vals if x is not None and y is not None]
            if not vals:
                continue
            mx = sum(x for x, _ in vals) / len(vals)
            my = sum(y for _, y in vals) / len(vals)
            better = sum(1 for x, y in vals if y > x + 1e-9)
            worse = sum(1 for x, y in vals if y < x - 1e-9)
            L.append(f"**{mname}**：{mx:.3f} → {my:.3f}（Δ{my - mx:+.3f}；逐题 ↑{better} / ↓{worse} / ={len(vals) - better - worse}）")
        L.append("")

        # 4) multi_hop 专项（ac 通过 + 误拒）
        mh = [q for q in ids if base[q].get("type") == "multi_hop"]
        if mh:
            ac_pairs = [(verdict(base[q], "answer_correctness"), verdict(ab[q], "answer_correctness")) for q in mh]
            ac_valid = [(x, y) for x, y in ac_pairs if x and y]
            fr = [(bool(base[q]["metrics"].get("false_refusal")), bool(ab[q]["metrics"].get("false_refusal"))) for q in mh]
            base_ac = sum(1 for x, _ in ac_valid if x == "pass")
            ab_ac = sum(1 for _, y in ac_valid if y == "pass")
            base_fr = sum(1 for x, _ in fr if x)
            ab_fr = sum(1 for _, y in fr if y)
            L.append(f"**multi_hop 专项（n={len(mh)}）**：ac 通过 {base_ac}→{ab_ac}；误拒 {base_fr}→{ab_fr}\n")
        L.append("")

    out_dir = REPO_ROOT / "eval_results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "p5_ablation_panel.md"
    out_path.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n面板已写出：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
