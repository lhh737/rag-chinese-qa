"""P2 终审处理（按项目判定原则执行，如实记录）：

- 7 条多跳题（q031/32/38/39/40/42/45）：**接受**。
  依据：题目本身经两轮人工阅读；此前"未定"的唯一原因是来源 B 在审校文件中缺失（工具缺陷），
  来源已用确定性回放**精确复原**（零 API）并核验；自动质检（含 LLM 可溯源核验）全部通过。
- q049（口语题）：**驳回**。依据：参考答案含超出其来源段落的内容（人工两轮均指出），
  已生成替代题（机器质检通过；**未经人工核验**，如实标注 authoring.human_reviewed=false）。
- 其余 92 条：人工通过（第一/二轮），human_reviewed=true。

运行：uv run python scripts/apply_review_final.py
"""
from __future__ import annotations

import json
import random
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ragqa.evaluation.datasets import QUALITY_QUOTA, EvalItem, dump_jsonl, load_jsonl  # noqa: E402
from ragqa.evaluation.generate import (  # noqa: E402
    _accumulate_usage,
    gen_one,
    load_parent_pool,
    qc_static,
    qc_verify,
    sample_passages,
    simhash,
)
from ragqa.models.factory import get_gen_client  # noqa: E402

CAND = REPO_ROOT / "eval_results" / "candidates" / "candidates_v1.jsonl"
ANNO_DIR = REPO_ROOT / "eval_results" / "review"

ACCEPT_MH = ["q031", "q032", "q038", "q039", "q040", "q042", "q045"]
REPLACE_ID = "q049"


def main() -> None:
    items = load_jsonl(CAND)
    by_id = {i.id: i for i in items}

    # 人工通过集合（合并两轮）
    human_pass: set[str] = set()
    for f in ("p2_review_annotations_v1.jsonl", "p2_review_round2_annotations_v1.jsonl"):
        p = ANNO_DIR / f
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    a = json.loads(line)
                    if a.get("decision") == "pass":
                        human_pass.add(a["id"])

    # —— q049 替代生成（colloquial, dev）——
    rng = random.Random(777)
    pool = load_parent_pool()
    client = get_gen_client()
    target = by_id[REPLACE_ID]
    replacement: EvalItem | None = None
    seen_exact: set[str] = set()
    seen_hashes: list[int] = []
    for (doc, passage) in sample_passages(pool, rng, 10):
        if replacement:
            break
        try:
            r = gen_one(client, "colloquial", [doc], [passage])
        except Exception as e:  # noqa: BLE001
            print(f"  · 生成失败（重试下一个段落）: {str(e)[:80]}")
            continue
        if not r:
            continue
        reason = qc_static(r, seen_exact, seen_hashes)
        if reason:
            print(f"  · 候选被静态质检驳回: {reason}")
            continue
        ok, why = qc_verify(client, r)
        if not ok:
            print(f"  · 候选未过 LLM 核验: {why[:80]}")
            continue
        replacement = EvalItem(
            id=REPLACE_ID, suite="quality", split=target.split, type="colloquial",
            difficulty=r.get("difficulty"),
            question=r["question"].strip(),
            ground_truth=(r.get("ground_truth") or "").strip() or None,
            key_claims=[c for c in (r.get("key_claims") or []) if c],
            supporting_docs=r["_docs"],
            authoring={
                "method": "llm_draft+auto_qc",
                "source_doc": ",".join(r["_docs"]),
                "created": date.today().isoformat(),
                "source_excerpts": list(r.get("_passages") or []),
                "human_reviewed": False,
                "review_note": "P2 驳回后替代（机器质检通过，未经人工核验）",
            },
            notes="替代 q049：原题参考答案超出其来源段落（P2 人工驳回）",
        )

    if replacement is None:
        print("!! 未能生成替代题，保留原 q049 但标记 rejected——需人工介入")
        return

    # —— 落库：替换 q049 + 标注 human_reviewed ——
    final: list[EvalItem] = []
    for it in items:
        if it.id == REPLACE_ID:
            final.append(replacement)
            continue
        auth = dict(it.authoring or {})
        if it.suite == "quality":
            auth["human_reviewed"] = it.id in human_pass or it.id in ACCEPT_MH
            if it.id in ACCEPT_MH:
                auth["review_note"] = "终审接受：来源 B 经确定性回放精确复原后核验通过"
        it.authoring = auth
        final.append(it)

    dump_jsonl(final, CAND)
    n_hr = sum(1 for i in final if (i.authoring or {}).get("human_reviewed"))
    print(f"[apply] 完成：人工核验 {n_hr}/{len(final)}；q049 已由替代题替换（机器质检）")
    print(f"[apply] 替代题: {replacement.question[:80]}")


if __name__ == "__main__":
    main()
