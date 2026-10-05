"""离线精确恢复 multi_hop 两个来源段落（零 API）——利用"生成采样是确定性的"这一性质。

原理：生成时的采样序列完全由 (seed=42, pool, plan 顺序, 函数调用序) 决定：
    factual: sample_passages(60) → multi_hop: sample_pairs(45) → colloquial: sample_passages(20)
    → out_of_scope（不消耗 rng）→ adversarial: sample_passages(30)
重放该序列即可得到当次运行用过的全部 (d1,p1,d2,p2)。
对每道 multi_hop 候选题，用其留存摘录前缀在重放结果中定位 p1 → 同时得到**精确的 p2**。
（背景：首版 authoring.source_excerpt 截断 900 字符导致 B 丢失；生成侧已修为 source_excerpts 列表。）

输出：eval_results/candidates/review_round3.jsonl（仅 --ids 指定题，来源全文）
运行：uv run python scripts/recover_sources.py --ids q031,q032,...
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ragqa.evaluation.datasets import QUALITY_QUOTA  # noqa: E402
from ragqa.evaluation.generate import load_parent_pool, sample_pairs, sample_passages  # noqa: E402

CAND_DIR = REPO_ROOT / "eval_results" / "candidates"
ANNO1 = REPO_ROOT / "eval_results" / "review" / "p2_review_annotations_v1.jsonl"
ANNO2 = REPO_ROOT / "eval_results" / "review" / "p2_review_round2_annotations_v1.jsonl"


def replay_tasks() -> tuple[list[tuple[str, list[str], list[str]]], list[tuple[str, list[str], list[str]]]]:
    """重放采样：返回 (全部任务, multi_hop 任务)。plan 与当次运行一致（×2，multi_hop ×3）。"""
    rng = random.Random(42)
    pool = load_parent_pool()
    tasks: list[tuple[str, list[str], list[str]]] = []
    mh_tasks: list[tuple[str, list[str], list[str]]] = []
    for qtype in ["factual", "multi_hop", "colloquial", "out_of_scope", "adversarial"]:
        q = QUALITY_QUOTA[qtype]
        n = q["total"] * (3 if qtype == "multi_hop" else 2)
        if qtype == "out_of_scope":
            tasks.extend([("out_of_scope", ["(库外语料)"], [""])] * n)
        elif qtype == "multi_hop":
            for d1, d2, p1, p2 in sample_pairs(pool, rng, n):
                t = ("multi_hop", [d1, d2], [p1, p2])
                tasks.append(t)
                mh_tasks.append(t)
        else:
            for d, p in sample_passages(pool, rng, n):
                tasks.append((qtype, [d], [p]))
    return tasks, mh_tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", required=True)
    args = parser.parse_args()
    ids = [x.strip() for x in args.ids.split(",") if x.strip()]

    items = {json.loads(l)["id"]: json.loads(l) for l in (CAND_DIR / "candidates_v1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    prior: dict[str, dict] = {}
    for p in (ANNO1, ANNO2):
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    a = json.loads(line)
                    prior[a["id"]] = prior.get(a["id"], {})
                    prior[a["id"]].update({k: a.get(k) for k in ("decision", "reason", "checks")})

    tasks, mh_tasks = replay_tasks()
    single_passages = [t for t in tasks if t[0] != "multi_hop" and len(t[2]) == 1 and t[2][0]]

    def _match(p: str, stored: str) -> bool:
        """强匹配：留存摘录是段首前缀（900 内）——要求全量前缀相等；
        若段短于留存，则要求留存以其开头且后接段落分隔/句末。"""
        if not p or not stored:
            return False
        n = len(stored)
        if len(p) >= n:
            return p[:n] == stored
        return stored.startswith(p) and (len(stored) == len(p) or stored[len(p) : len(p) + 2] == "\n\n")

    def find_by_prefix(cands, it):
        stored = ((it.get("authoring") or {}).get("source_excerpt") or "")
        if not stored:
            return None
        hits = [t for t in cands if any(_match(p, stored) for p in t[2] if p)]
        # supporting_docs 交叉锁定（生成时 _docs 与配对严格一致）
        sd = set(it.get("supporting_docs") or [])
        matched = [t for t in hits if set(t[1]) == sd]
        if len(matched) == 1:
            return matched[0]
        if len(matched) > 1:
            print(f"  ⚠ {it['id']} supporting_docs 锁定后仍有 {len(matched)} 个候选，取第一个")
            return matched[0]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            print(f"  ⚠ {it['id']} 强匹配命中 {len(hits)} 个任务且 supporting_docs 无法区分：{[t[1] for t in hits]}")
            return hits[0]
        return None

    out_rows = []
    for i in ids:
        it = items.get(i)
        if not it:
            print(f"  ! 缺 {i}")
            continue
        if it["type"] == "multi_hop":
            t = find_by_prefix(mh_tasks, it)
            if t is None:
                print(f"  ! {i} 前缀未命中（无法恢复）")
                continue
            d1, d2 = t[1]
            p1, p2 = t[2]
            excerpt = f"【来源 A·{d1}】\n{p1}\n\n【来源 B·{d2}】\n{p2}"
        else:
            t = find_by_prefix(single_passages, it)
            if t is None:
                print(f"  ! {i} 前缀未命中")
                continue
            excerpt = t[2][0]
        a = prior.get(i, {})
        out_rows.append({
            "id": i, "suite": "quality", "type": it["type"], "split": it["split"],
            "question": it["question"], "ground_truth": it.get("ground_truth"),
            "key_claims": it.get("key_claims") or [], "supporting_docs": it.get("supporting_docs") or [],
            "expected_behavior": it.get("expected_behavior"), "authoring": it.get("authoring") or {},
            "source_excerpt": excerpt, "match_score": None,
            "prior_decision": a.get("decision") or "未判定", "prior_reason": a.get("reason") or "",
        })

    out = CAND_DIR / "review_round3.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[recover] 第三轮（8 条精确来源）→ {out}")
    for r in out_rows:
        ab = f"A/B 双来源" if "【来源 B" in r["source_excerpt"] else "单来源"
        print(f"  {r['id']}: {ab}，摘录 {len(r['source_excerpt'])} 字符")


if __name__ == "__main__":
    main()
