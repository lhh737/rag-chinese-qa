"""构建 P4 盲标文件（judge 校准用）：60 题 dev 全集 + 三段分配（示例/调参/测量 20/20/20）。

- 数据源：P3 基线 run（20261005-001）的 trace（question/contexts/answer）+ golden_v1（GT/claims）
- **不含任何 judge 结论**（盲标纪律）；适用 criteria 由 expected_behavior 决定（与 runner 一致）
- 三段分配按题型分层（大题型优先，容量均衡）→ 恰好 20/20/20；分配表单独存放：
  调参期间**不得查看测量段**（设计 §4.5）

输出：
  eval_results/calibration/blind_label_v1.jsonl     （查看器模式②逐题盲标）
  eval_results/calibration/calibration_segments.json（id → segment 映射与纪律说明）
运行：uv run python scripts/build_blind_file.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ragqa.evaluation.datasets import load_jsonl  # noqa: E402
from ragqa.evaluation.runner import criteria_for  # noqa: E402

RUN = REPO_ROOT / "eval_results" / "20261005-001"
OUT_DIR = REPO_ROOT / "eval_results" / "calibration"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = [i for i in load_jsonl(REPO_ROOT / "data/golden/golden_v1.jsonl") if i.split == "dev"]
    results = {json.loads(l)["id"]: json.loads(l) for l in (RUN / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    traces = {json.loads(l)["trace_id"]: json.loads(l) for l in (RUN / "traces.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}

    # 三段分配（题型分层 + 容量均衡 → 恰好 20/20/20）
    by_type: dict[str, list] = defaultdict(list)
    for it in items:
        by_type[it.type].append(it)
    caps = [20, 20, 20]
    seg_of: dict[str, int] = {}
    for t in sorted(by_type, key=lambda x: -len(by_type[x])):
        for it in sorted(by_type[t], key=lambda x: x.id):
            seg = caps.index(max(caps))
            caps[seg] -= 1
            seg_of[it.id] = seg
    assert caps == [0, 0, 0], f"三段分配不均: {caps}"

    rows = []
    for it in sorted(items, key=lambda x: x.id):
        r = results.get(it.id)
        if not r:
            print(f"  ! {it.id} 无运行结果（跳过）")
            continue
        trace = traces[r["trace_id"]]
        ctx = trace.get("context") or {}
        contexts = [
            {"rank": i + 1, "source": (ctx.get("sources") or [""] * len(ctx.get("texts") or []))[i] if i < len(ctx.get("sources") or []) else "",
             "text": txt}
            for i, txt in enumerate(ctx.get("texts") or [])
        ]
        rows.append({
            "id": it.id, "type": it.type, "question": it.question,
            "ground_truth": it.ground_truth, "key_claims": it.key_claims,
            "supporting_docs": it.supporting_docs,
            "expected_behavior": it.expected_behavior,
            "contexts": contexts,
            "answer": (trace.get("generation") or {}).get("answer", ""),
            "criteria": criteria_for(it),
        })

    blind_path = OUT_DIR / "blind_label_v1.jsonl"
    with open(blind_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    seg_path = OUT_DIR / "calibration_segments.json"
    seg_path.write_text(json.dumps({
        "note": "三段式纪律：示例段(0)=few-shot 来源，不参与测量；调参段(1)=仅用于调 prompt；测量段(2)=最终 TPR/TNR，调参期间不得查看",
        "segments": {i: ("示例段" if s == 0 else "调参段" if s == 1 else "测量段") for i, s in seg_of.items()},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter
    print(f"[blind] {len(rows)} 题 → {blind_path}")
    print(f"[blind] 三段分布: {dict(Counter('示例' if s==0 else '调参' if s==1 else '测量' for s in seg_of.values()))}")


if __name__ == "__main__":
    main()
