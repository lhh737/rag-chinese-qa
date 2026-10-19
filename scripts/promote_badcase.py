"""badcase → 回归资产 / 候选评测题（默认只写回归目录，不改 golden）。

用法：
    uv run python scripts/promote_badcase.py --run-id 20261013-003
    uv run python scripts/promote_badcase.py \
        --input eval_results/20261013-003/attribution.jsonl \
        --out-dir eval_results/regression

输入支持两类行：
1) 归因产物（含 id / attribution / suggestion）：已有题 → 打“回归标签”；
2) 新 badcase（含 question / ground_truth / key_claims / supporting_docs / type）：
   走 去重 → 防泄漏 → 配额 三道门，通过后进入 promoted_v1.jsonl（待人工审校）。

硬约束：
- 不自动修改 `data/golden/`；入集必须人工审批 + CHANGELOG + hash；
- 去重/防泄漏阈值复用 P1 出题模块（SimHash ≤3、5-gram 重合 ≤0.7），保持口径一致；
- 没有真实新 badcase 时，脚本只产出回归标签，不虚构新题。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ragqa.evaluation.generate import (  # noqa: E402
    LEAK_RATIO_CAP,
    SIMHASH_MAX_DIST,
    hamming,
    leak_ratio,
    simhash,
)

DEFAULT_INPUT = REPO_ROOT / "eval_results" / "20261013-003" / "attribution.jsonl"
DEFAULT_GOLDEN = REPO_ROOT / "data" / "golden" / "golden_v1.jsonl"
DEFAULT_SAFETY = REPO_ROOT / "data" / "golden" / "safety_v1.jsonl"
DEFAULT_OUT = REPO_ROOT / "eval_results" / "regression"


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "（无）"
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    parser.add_argument("--safety", default=str(DEFAULT_SAFETY))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--run-id", default="20261013-003")
    parser.add_argument("--max-per-type", type=int, default=None,
                        help="新题每个题型最多接收 N 条；缺省不设上限（仅报告配额）")
    args = parser.parse_args()

    existing = _rows(Path(args.golden)) + _rows(Path(args.safety))
    existing_by_id = {item.get("id"): item for item in existing if item.get("id")}
    existing_questions = {_norm(item.get("question", "")) for item in existing}
    existing_hashes = [simhash(item.get("question", "")) for item in existing if item.get("question")]
    quota = Counter(item.get("type") for item in existing if item.get("suite") == "quality")

    input_rows = _rows(Path(args.input))
    if not input_rows:
        print(f"✗ 输入为空或不存在：{args.input}")
        return 2

    regression: list[dict] = []
    promoted: list[dict] = []
    rejected: list[dict] = []
    accepted_by_type: Counter = Counter()

    for row in input_rows:
        qid = row.get("id")
        # 分支 A：已有题（基线失败项）→ 回归标签，不重复入集
        if qid in existing_by_id:
            failure = row.get("attribution") or "未分类"
            if failure == "无问题":
                continue
            regression.append({
                "id": qid,
                "source_run": args.run_id,
                "type": existing_by_id[qid].get("type") or existing_by_id[qid].get("safety_type"),
                "question": existing_by_id[qid].get("question"),
                "failure_class": failure,
                "secondary": row.get("secondary", []),
                "suggestion": row.get("suggestion", ""),
                "status": "existing_regression",
            })
            continue

        # 分支 B：新 badcase → 三道门
        question = row.get("question") or ""
        if not question.strip():
            rejected.append({"id": qid, "reason": "missing_question"})
            continue
        qtype = row.get("type")
        if row.get("expected_behavior") != "refuse" and not (row.get("ground_truth") and row.get("key_claims")):
            rejected.append({"id": qid, "reason": "missing_ground_truth_or_key_claims"})
            continue
        if not row.get("supporting_docs") and row.get("expected_behavior") != "refuse":
            rejected.append({"id": qid, "reason": "missing_supporting_docs"})
            continue

        norm_q = _norm(question)
        if norm_q in existing_questions:
            rejected.append({"id": qid, "reason": "duplicate_exact"})
            continue
        h = simhash(question)
        if any(hamming(h, x) <= SIMHASH_MAX_DIST for x in existing_hashes):
            rejected.append({"id": qid, "reason": "duplicate_simhash"})
            continue
        source = (row.get("authoring") or {}).get("source_excerpt", "")
        lr = leak_ratio(question, source) if source else 0.0
        if source and lr > LEAK_RATIO_CAP:
            rejected.append({"id": qid, "reason": f"leakage:{lr:.2f}"})
            continue
        if args.max_per_type is not None and qtype and accepted_by_type[qtype] >= args.max_per_type:
            rejected.append({"id": qid, "reason": f"quota_full:{qtype}"})
            continue

        accepted_by_type[qtype] += 1
        promoted.append({
            **row,
            "promotion": {
                "source_run": args.run_id,
                "dedup": "passed",
                "leak_ratio": round(lr, 4),
                "proposed_at": datetime.now().strftime("%Y-%m-%d"),
                "status": "pending_human_review",
            },
        })

    out_dir = Path(args.out_dir)
    _write_jsonl(regression, out_dir / "regression_v1.jsonl")
    _write_jsonl(promoted, out_dir / "promoted_v1.jsonl")

    reason_counts = Counter(item["reason"] for item in rejected)
    fail_counts = Counter(item["failure_class"] for item in regression)
    report = [
        "# badcase 回流报告（dry-run）",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 输入：`{Path(args.input).relative_to(REPO_ROOT).as_posix()}`（{len(input_rows)} 行）",
        f"- 已有集：`{Path(args.golden).relative_to(REPO_ROOT).as_posix()}` + "
        f"`{Path(args.safety).relative_to(REPO_ROOT).as_posix()}`（{len(existing)} 题）",
        f"- 阈值：SimHash ≤{SIMHASH_MAX_DIST} 判近重复；5-gram 重合 >{LEAK_RATIO_CAP} 判泄漏。",
        "- **本脚本不改 `data/golden/`；入集必须人工审批 + CHANGELOG + hash。**",
        "",
        "## 1. 回归标签（已有失败题，不重复入集）",
        "",
        f"共 {len(regression)} 条；失败类分布：",
        "",
        _md_table(["失败类", "n"], [[k, str(v)] for k, v in fail_counts.most_common()]),
        "",
        f"明细：`{(out_dir / 'regression_v1.jsonl').relative_to(REPO_ROOT).as_posix()}`",
        "",
        "## 2. 新题候选（三道门）",
        "",
        f"- 接收：**{len(promoted)}** 条（待人工审校，未入集）",
        f"- 拒绝：{len(rejected)} 条",
        "",
        _md_table(["拒绝原因", "n"], [[k, str(v)] for k, v in reason_counts.most_common()]),
        "",
        f"明细：`{(out_dir / 'promoted_v1.jsonl').relative_to(REPO_ROOT).as_posix()}`",
        "",
        "## 3. 现有配额（供下一版决策）",
        "",
        _md_table(["题型", "现有题数"], [[k, str(v)] for k, v in sorted(quota.items())]),
        "",
        "## 4. 下一步",
        "",
        "1. 对 `promoted_v1.jsonl` 做人工审校（题合理 / 答案可溯源 / 支撑文档正确）；",
        "2. 审校通过后，按 `data/golden/CHANGELOG.md` 的格式记版本 + hash；",
        "3. 重新跑健康度检查和基线，确认新题不泄漏、不破坏配额；",
        "4. 回归标签可直接接入 CI 的回归集（不改 golden 版本）。",
        "",
        "## 5. 诚实边界",
        "",
        "- 本次输入是**已有失败归因**，不是生产流量；因此新增题通常为 0，产出的是回归标签。",
        "- 新题通路的去重/防泄漏/配额门已实现；有真实新 badcase 时，把 `--input` 换成新数据即可。",
        "- 不要把“回归标签”说成“新评测集”；两者在版本管理上是不同资产。",
        "",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "promotion_report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"✓ 回归标签 {len(regression)} 条 → {out_dir / 'regression_v1.jsonl'}")
    print(f"✓ 新题候选 {len(promoted)} 条 / 拒绝 {len(rejected)} 条 → {out_dir / 'promoted_v1.jsonl'}")
    print(f"✓ 报告 → {out_dir / 'promotion_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
