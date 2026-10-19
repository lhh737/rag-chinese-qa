"""judge 迁移抽检：生成 CSV、合并人工判定、计算 judge–human 一致性。

用法：
    uv run python scripts/apply_judge_spotcheck.py --init      # 生成空白 CSV
    uv run python scripts/apply_judge_spotcheck.py --check     # 查看还有多少条没填
    uv run python scripts/apply_judge_spotcheck.py             # 合并并生成报告

CSV 只需填两列：
    human_verdict：pass / fail（留空 = 未填）
    note：分歧理由或“原人工标签 X，现判 Y”

口径：
- 以你对照材料后的独立判断为准，不要为了和 judge 一致而改；
- judge–human 一致性是“judge 在新域是否可用”的证据，不是人类 κ。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from calibrate_judge import metrics  # noqa: E402

DEFAULT_CSV = REPO_ROOT / "data" / "business" / "judge_spotcheck_v1.csv"
DEFAULT_JSONL = REPO_ROOT / "data" / "business" / "judge_spotcheck_v1.jsonl"
DEFAULT_REPORT = REPO_ROOT / "data" / "business" / "judge_spotcheck_v1_report.md"

FIELDS = ["case_id", "id", "type", "question", "criterion",
          "judge_verdict", "judge_reason", "human_verdict", "note"]


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_csv(path: Path) -> list[dict]:
    for enc in ("utf-8-sig", "gbk"):
        try:
            with path.open("r", encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"无法解码 CSV：{path}（请另存为 UTF-8）")


def init_csv(jsonl_path: Path, csv_path: Path) -> None:
    rows = _rows(jsonl_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            for criterion, judge in (r.get("judge") or {}).items():
                w.writerow({
                    "case_id": r.get("case_id", ""),
                    "id": r.get("id", ""),
                    "type": r.get("type", ""),
                    "question": r.get("question", ""),
                    "criterion": criterion,
                    "judge_verdict": (judge or {}).get("verdict", ""),
                    "judge_reason": ((judge or {}).get("reason") or "")[:500],
                    "human_verdict": "",
                    "note": "",
                })
    print(f"✓ 空白 CSV → {csv_path}")
    print(f"  行数：{sum(len(r.get('judge') or {}) for r in rows)}（20 题 × 适用 criterion）")


def check(csv_path: Path) -> None:
    rows = _read_csv(csv_path)
    filled = [r for r in rows if (r.get("human_verdict") or "").strip()]
    print(f"已填 {len(filled)} / {len(rows)}；剩余 {len(rows) - len(filled)} 条")
    for r in rows:
        if not (r.get("human_verdict") or "").strip():
            print(f"  - {r['case_id']}  {r['criterion']}  judge={r['judge_verdict']}")


def _val(x):
    return x.get("value") if isinstance(x, dict) else x


def _fmt(x):
    return "—" if x is None else f"{x:.3f}"


def merge(csv_path: Path, jsonl_path: Path, report_path: Path) -> int:
    rows = _read_csv(csv_path)
    by_case: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        v = (r.get("human_verdict") or "").strip().lower()
        if v not in ("", "pass", "fail"):
            print(f"✗ {r['case_id']} {r['criterion']}: human_verdict 必须是 pass/fail/空，实际 {v!r}")
            return 2
        by_case[r["case_id"]].append(r)

    items = _rows(jsonl_path)
    pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    disagreements = []
    total = 0
    filled = 0
    for item in items:
        case_id = item.get("case_id", "")
        notes = []
        for criterion, judge in (item.get("judge") or {}).items():
            total += 1
            match = next((r for r in by_case.get(case_id, []) if r["criterion"] == criterion), None)
            human = (match.get("human_verdict") or "").strip().lower() if match else ""
            item.setdefault("human_verdicts", {})[criterion] = human or None
            if match and (match.get("note") or "").strip():
                notes.append(f"{criterion}: {match['note'].strip()}")
            if human in ("pass", "fail"):
                filled += 1
                j = (judge or {}).get("verdict")
                if j in ("pass", "fail"):
                    pairs[criterion].append((human, j))
                    if human != j:
                        disagreements.append({
                            "case_id": case_id, "criterion": criterion,
                            "human": human, "judge": j,
                            "judge_reason": (judge or {}).get("reason", ""),
                            "note": (match.get("note") or "").strip() if match else "",
                        })
        item["human_note"] = "；".join(notes)
        item["reviewed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if notes else item.get("reviewed_at")

    jsonl_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items) + "\n", encoding="utf-8")

    all_pairs = [p for ps in pairs.values() for p in ps]
    overall = metrics(all_pairs) if all_pairs else None

    def row(name: str, m: dict | None, n_pending: int) -> str:
        if not m:
            return f"| {name} | 0 | — | — | — | — | — | {n_pending} |"
        conf = m["confusion"]
        return (f"| {name} | {m['n']} | {_fmt(_val(m.get('agreement')))} | {_fmt(m.get('kappa'))} | "
                f"{_fmt(_val(m.get('tpr')))} | {_fmt(_val(m.get('tnr')))} | "
                f"{conf['tp']}/{conf['fn']}/{conf['tn']}/{conf['fp']} | {n_pending} |")

    per_crit_pending = {}
    for c in ("answer_correctness", "faithfulness", "citation_accuracy", "oos_refusal"):
        crit_rows = [r for r in rows if r["criterion"] == c]
        per_crit_pending[c] = sum(1 for r in crit_rows if not (r.get("human_verdict") or "").strip())

    lines = [
        "# 第二域 judge 迁移抽检报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- CSV：`{csv_path.relative_to(REPO_ROOT).as_posix()}`",
        f"- 已填：{filled} / {total}；未填：{total - filled}",
        "",
    ]
    if total - filled:
        lines += [f"> ⚠ 还有 {total - filled} 条未填；下表只统计已填部分，**不作最终结论**。", ""]
    lines += [
        "## judge–human 一致性",
        "",
        "| criterion | n | agreement | κ | TPR | TNR | tp/fn/tn/fp | 未填 |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
        row("answer_correctness", metrics(pairs.get("answer_correctness", [])), per_crit_pending["answer_correctness"]),
        row("faithfulness", metrics(pairs.get("faithfulness", [])), per_crit_pending["faithfulness"]),
        row("citation_accuracy", metrics(pairs.get("citation_accuracy", [])), per_crit_pending["citation_accuracy"]),
        row("oos_refusal", metrics(pairs.get("oos_refusal", [])), per_crit_pending["oos_refusal"]),
        row("**overall**", overall, total - filled),
        "",
        "> 以人工判定为参照：TPR = 人工 fail 中 judge 也 fail；TNR = 人工 pass 中 judge 也 pass。",
        "> 这是“judge 在新域是否可用”的证据，不是人类标注者间 κ。",
        "",
        "## 分歧",
        "",
    ]
    if disagreements:
        lines += ["| case | criterion | human | judge | judge 理由 | 备注 |", "|---|---|---|---|---|---|"]
        for d in disagreements:
            lines.append(f"| {d['case_id']} | {d['criterion']} | {d['human']} | {d['judge']} | "
                         f"{d['judge_reason'][:160]} | {d['note'][:120]} |")
    else:
        lines.append("无分歧。" if filled == total else "尚未填完。")
    lines += ["", "## 下一步", "", "1. 填完剩余项后重新运行本脚本；", "2. 把结论回填到 `docs/transferability.md` §6；", "3. 若 judge 在某些 criterion 明显偏低，先改判定规范再改 judge prompt。"]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ 已合并 → {jsonl_path}")
    print(f"✓ 报告 → {report_path}")
    print(f"  已填 {filled}/{total}" + (f"；仍未填 {total - filled}" if total - filled else "；全部完成"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--jsonl", default=str(DEFAULT_JSONL))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--init", action="store_true", help="从 JSONL 生成空白 CSV")
    parser.add_argument("--check", action="store_true", help="只统计未填项")
    args = parser.parse_args()

    csv_path, jsonl_path, report_path = Path(args.csv), Path(args.jsonl), Path(args.report)
    if args.init:
        init_csv(jsonl_path, csv_path)
        return 0
    if args.check:
        check(csv_path)
        return 0
    return merge(csv_path, jsonl_path, report_path)


if __name__ == "__main__":
    raise SystemExit(main())
