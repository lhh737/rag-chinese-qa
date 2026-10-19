"""生成第二域 judge 迁移抽检材料（20 条：8 失败 + 12 通过对照）。

用法：
    uv run python scripts/build_judge_spotcheck.py --run-id 20261019-001

输出：
    data/business/judge_spotcheck_v1.jsonl
    data/business/judge_spotcheck_v1.md

说明：
- 这是“judge 在新域是否还能用”的人工抽检材料，不是自动指标；
- human_verdicts 留空，由人工逐 criterion 填 pass/fail；
- 失败项覆盖全部 8 条；通过项按题型分层选 12 条，含库外/对抗对照。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "business"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _verdicts(row: dict) -> dict[str, str]:
    return {k: (v or {}).get("verdict", "—") for k, v in (row.get("judge") or {}).items()}


def _is_fail(row: dict) -> bool:
    return any(v == "fail" for v in _verdicts(row).values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="20261019-001")
    args = parser.parse_args()

    run_dir = REPO_ROOT / "eval_results" / args.run_id
    results = _rows(run_dir / "results.jsonl")
    traces = {r["trace_id"]: r for r in _rows(run_dir / "traces.jsonl")}
    items = {r["id"]: r for r in _rows(REPO_ROOT / "data" / "business" / "business_v1.jsonl")}
    items.update({r["id"]: r for r in _rows(REPO_ROOT / "data" / "golden" / "golden_v1.jsonl")})
    items.update({r["id"]: r for r in _rows(REPO_ROOT / "data" / "golden" / "safety_v1.jsonl")})

    failed = [r for r in results if _is_fail(r)]
    passed = [r for r in results if not _is_fail(r)]

    # 12 条通过对照：按题型分层，优先覆盖库外/对抗
    desired = {"out_of_scope": 2, "adversarial": 2, "factual": 3, "multi_hop": 3, "colloquial": 2}
    picked: list[dict] = []
    for qtype, n in desired.items():
        subset = [r for r in passed if r.get("type") == qtype]
        picked += subset[:n]
    # 不足 12 条时用其他通过项补齐
    for r in passed:
        if len(picked) >= 12:
            break
        if r not in picked:
            picked.append(r)
    selected = failed + picked[:12]

    rows = []
    for r in selected:
        item = items.get(r["id"], {})
        trace = traces.get(r.get("trace_id", ""), {})
        ctx = (trace.get("context") or {})
        texts = list(ctx.get("texts") or [])
        sources = list(ctx.get("sources") or [])
        answer = (trace.get("generation") or {}).get("answer", "")
        judge = {
            k: {
                "verdict": (v or {}).get("verdict"),
                "reason": (v or {}).get("reason"),
                "evidence": (v or {}).get("evidence"),
            }
            for k, v in (r.get("judge") or {}).items()
        }
        rows.append({
            "case_id": f'{args.run_id}:{r["id"]}',
            "id": r["id"],
            "type": r.get("type"),
            "question": item.get("question") or r.get("question"),
            "ground_truth": item.get("ground_truth"),
            "key_claims": item.get("key_claims", []),
            "answer": answer,
            "contexts": [
                {"n": i + 1, "source": sources[i] if i < len(sources) else "", "text": t}
                for i, t in enumerate(texts[:5])
            ],
            "judge": judge,
            "human_verdicts": {k: None for k in judge},
            "human_note": "",
            "is_failure": _is_fail(r),
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUT_DIR / "judge_spotcheck_v1.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")

    # Markdown 表单
    lines = [
        "# 第二域 judge 迁移抽检（20 条）",
        "",
        f"> 运行：`{args.run_id}` · 失败 {len(failed)} 条全部入选 + 通过对照 {len(picked[:12])} 条",
        "> 逐 criterion 填 `pass/fail`；分歧写一句理由。抽检完成后把结果回填到 JSONL 的 `human_verdicts`。",
        "",
    ]
    for row in rows:
        lines += [
            f"## {row['id']} · {row['type']} · judge={'FAIL' if row['is_failure'] else 'PASS'}",
            "",
            f"**Q**：{row['question']}",
            "",
            f"**GT**：{row.get('ground_truth') or '（库外）'}",
            "",
            f"**Answer**：{row['answer']}",
            "",
        ]
        for c in row["contexts"]:
            text = re.sub(r"\s+", " ", c["text"])[:500]
            lines.append(f"- [{c['n']}]（{c['source']}）{text}")
        lines += ["", "**Judge 理由**："]
        for k, v in row["judge"].items():
            lines.append(f"- `{k}` = **{v.get('verdict')}**：{(v.get('reason') or '')[:300]}")
        crits = "；".join(f"{k} [ ]pass [ ]fail" for k in row["judge"])
        lines += ["", f"**你的判定**：{crits}", "", "**备注**：", "", "---", ""]
    (OUT_DIR / "judge_spotcheck_v1.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"✓ {len(rows)} 条抽检 → {jsonl_path}")
    print(f"✓ 表单 → {OUT_DIR / 'judge_spotcheck_v1.md'}")
    print(f"  失败 {len(failed)} + 通过对照 {len(picked[:12])}")
    for row in rows:
        print(f"  {'FAIL' if row['is_failure'] else 'pass'}  {row['id']}  {row['type']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
