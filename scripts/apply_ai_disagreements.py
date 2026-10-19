"""AI 第二标注分歧裁决：生成 CSV、合并裁决、汇总。

用法：
    uv run python scripts/apply_ai_disagreements.py --init
    uv run python scripts/apply_ai_disagreements.py --check
    uv run python scripts/apply_ai_disagreements.py

CSV 只需填两列：
    adjudication：ai_over_strict / human_missed / unclear（留空 = 未填）
    note：一句话理由

口径：
- ai_over_strict：AI 判 fail、人工判 pass，复核后认为人工 pass 正确；
- human_missed：复核后认为 AI 判 fail 正确，原人工 pass 是漏判；
- unclear：判定规范本身有歧义，需改 guideline。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "eval_results" / "calibration" / "ai_second_disagreements_v1.csv"
DEFAULT_JSONL = REPO_ROOT / "eval_results" / "calibration" / "ai_second_disagreements_v1.jsonl"
DEFAULT_REPORT = REPO_ROOT / "eval_results" / "calibration" / "reports" / "P4b_ai_disagreements_adjudicated.md"

FIELDS = ["id", "criterion", "human", "ai", "ai_reason", "human_reason", "adjudication", "note"]


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
            w.writerow({
                "id": r.get("id", ""),
                "criterion": r.get("criterion", ""),
                "human": r.get("human", ""),
                "ai": r.get("ai", ""),
                "ai_reason": (r.get("ai_reason") or "")[:500],
                "human_reason": (r.get("human_reason") or "")[:500],
                "adjudication": "",
                "note": "",
            })
    print(f"✓ 空白 CSV → {csv_path}（{len(rows)} 条分歧）")


def check(csv_path: Path) -> None:
    rows = _read_csv(csv_path)
    filled = [r for r in rows if (r.get("adjudication") or "").strip()]
    print(f"已填 {len(filled)} / {len(rows)}；剩余 {len(rows) - len(filled)} 条")
    for r in rows:
        if not (r.get("adjudication") or "").strip():
            print(f"  - {r['id']}  {r['criterion']}  human={r['human']} ai={r['ai']}")


def merge(csv_path: Path, jsonl_path: Path, report_path: Path) -> int:
    rows = _read_csv(csv_path)
    valid = {"", "ai_over_strict", "human_missed", "unclear"}
    for r in rows:
        v = (r.get("adjudication") or "").strip()
        if v not in valid:
            print(f"✗ {r['id']} {r['criterion']}: adjudication 必须是 ai_over_strict/human_missed/unclear/空，实际 {v!r}")
            return 2

    by_key = {(r["id"], r["criterion"]): r for r in rows}
    items = _rows(jsonl_path)
    for item in items:
        r = by_key.get((item.get("id"), item.get("criterion")))
        if r:
            item["adjudication"] = (r.get("adjudication") or "").strip() or None
            item["adjudication_note"] = (r.get("note") or "").strip()
    jsonl_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items) + "\n", encoding="utf-8")

    filled = [r for r in rows if (r.get("adjudication") or "").strip()]
    counts = Counter((r.get("adjudication") or "").strip() for r in filled)
    by_crit = defaultdict(Counter)
    for r in filled:
        by_crit[r["criterion"]][(r.get("adjudication") or "").strip()] += 1

    lines = [
        "# P4b · AI 第二标注分歧裁决",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 已裁决：{len(filled)} / {len(rows)}",
        "",
        "## 汇总",
        "",
        "| adjudication | n | 含义 |",
        "|---|---:|---|",
        f"| ai_over_strict | {counts.get('ai_over_strict', 0)} | AI 过严，人工 pass 正确 |",
        f"| human_missed | {counts.get('human_missed', 0)} | AI fail 正确，原人工 pass 是漏判 |",
        f"| unclear | {counts.get('unclear', 0)} | 规范歧义，需改 guideline |",
        f"| 未填 | {len(rows) - len(filled)} | — |",
        "",
        "## 按 criterion",
        "",
        "| criterion | ai_over_strict | human_missed | unclear |",
        "|---|---:|---:|---:|",
    ]
    for c in sorted(by_crit):
        lines.append(f"| {c} | {by_crit[c].get('ai_over_strict', 0)} | {by_crit[c].get('human_missed', 0)} | {by_crit[c].get('unclear', 0)} |")
    lines += ["", "## 明细", "", "| id | criterion | human | AI | 裁决 | 备注 |", "|---|---|---|---|---|---|"]
    for r in filled:
        lines.append(f"| {r['id']} | {r['criterion']} | {r['human']} | {r['ai']} | {r.get('adjudication')} | {(r.get('note') or '')[:160]} |")
    lines += [
        "",
        "## 读法",
        "",
        "- `ai_over_strict` 多 → AI 第二标注可作为“严格预筛”，但会增加人工复核量；",
        "- `human_missed` 多 → 原人工盲标有漏判，需回看校准集；",
        "- `unclear` 多 → 先改判定规范，再改 judge/annotator prompt。",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ 已合并 → {jsonl_path}")
    print(f"✓ 报告 → {report_path}")
    print(f"  已裁决 {len(filled)}/{len(rows)}；ai_over_strict={counts.get('ai_over_strict',0)} "
          f"human_missed={counts.get('human_missed',0)} unclear={counts.get('unclear',0)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--jsonl", default=str(DEFAULT_JSONL))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--check", action="store_true")
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
