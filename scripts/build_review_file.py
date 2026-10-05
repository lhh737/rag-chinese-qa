"""构建 P2 人工审校文件（离线，零 API）。

背景与两次迭代（诚实记录）：
- v1 的问题：来源摘录被截断（900 字符）+ multi_hop 只展示论文 A → 引发"内容缺失"类误驳回；
  这是**测量工具缺陷而非候选题缺陷**（P2 第一轮的实际教训）。
- v2（本脚本）：来源段落**全文**展示；multi_hop 分 A/B 双来源；并支持 `--ids` 生成
  "第二轮精审"文件（只含被标记的题），来源通过**前缀精确匹配**回溯父块原文取全。

输出：
  默认     → eval_results/candidates/review_v1.jsonl（全量审校）
  --ids …  → eval_results/candidates/review_round2.jsonl（第二轮精审：带上一轮判定与理由）

运行：uv run python scripts/build_review_file.py [--ids q003,q036,...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ragqa.utils.paths import resolve_path  # noqa: E402

CAND_DIR = REPO_ROOT / "eval_results" / "candidates"
ANNO_PATH = REPO_ROOT / "eval_results" / "review" / "p2_review_annotations_v1.jsonl"


def ngrams(text: str, n: int = 5) -> set[str]:
    t = "".join(ch for ch in text if not ch.isspace())
    return {t[i : i + n] for i in range(max(len(t) - n + 1, 0))}


def load_parents_by_doc() -> dict[str, list[str]]:
    raw = json.loads(resolve_path("faiss_db/parent_store.json").read_text(encoding="utf-8"))
    by_doc: dict[str, list[str]] = {}
    for blob in raw.values():
        meta = blob.get("metadata") or {}
        did = meta.get("doc_id") or ""
        if did:
            by_doc.setdefault(did, []).append(blob.get("page_content") or "")
    return by_doc


def recover_prefix(by_doc: dict[str, list[str]], docs: list[str], prefix: str, n: int = 120) -> str | None:
    """用已存摘录的前缀精确匹配父块原文（生成时摘录就是父块开头）。"""
    key = prefix[:n]
    if not key:
        return None
    for did in docs:
        for p in by_doc.get(did, []):
            if p.startswith(key) or p[:n] == key:
                return p
    # 兜底：全库前缀匹配
    for did in by_doc:
        for p in by_doc[did]:
            if p.startswith(key):
                return p
    return None


def best_match(by_doc: dict[str, list[str]], doc: str, query_text: str) -> tuple[str, float]:
    q = ngrams(query_text)
    best, score = "", 0.0
    for p in by_doc.get(doc, []):
        pg = ngrams(p)
        if not pg:
            continue
        s = len(q & pg) / max(len(q), 1)
        if s > score:
            best, score = p, s
    return best, round(score, 3)


def base_row(item: dict, suite: str) -> dict:
    return {
        "id": item["id"], "suite": suite, "type": item.get("type") or item.get("safety_type"),
        "split": item.get("split"), "question": item["question"],
        "ground_truth": item.get("ground_truth"), "key_claims": item.get("key_claims") or [],
        "supporting_docs": item.get("supporting_docs") or [],
        "expected_behavior": item.get("expected_behavior"),
        "authoring": item.get("authoring") or {},
    }


def build_full(by_doc) -> None:
    out_rows: list[dict] = []
    for suite_file, suite in (("candidates_v1.jsonl", "quality"), ("safety_v1_candidates.jsonl", "safety")):
        src = CAND_DIR / suite_file
        if not src.is_file():
            continue
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            row = base_row(item, suite)
            if suite == "safety":
                st = item.get("safety_type")
                row["source_excerpt"] = (
                    f"（文档注入题：测试语料见 data/safety_corpus/，marker={item.get('injection_marker')}）"
                    if st == "injection_doc" else "（安全题：人工构造，无来源段落）"
                )
                row["match_score"] = None
            else:
                stored = (item.get("authoring") or {}).get("source_excerpt") or ""
                if stored:
                    docs = row["supporting_docs"]
                    parts = stored.split("\n\n")
                    p1 = recover_prefix(by_doc, docs, parts[0]) or parts[0]
                    if len(parts) > 1 and len(docs) > 1:
                        p2 = best_match(by_doc, docs[1], (item.get("question") or "") + (item.get("ground_truth") or ""))[0]
                        row["source_excerpt"] = f"【来源 A·{docs[0]}】\n{p1}\n\n【来源 B·{docs[1]}】\n{p2}"
                    else:
                        row["source_excerpt"] = p1
                elif row["supporting_docs"]:
                    row["source_excerpt"], row["match_score"] = best_match(
                        by_doc, row["supporting_docs"][0], (item.get("question") or "") + (item.get("ground_truth") or "")
                    )[0], None
                else:
                    row["source_excerpt"] = "（库外题：无来源）"
                row["match_score"] = None
            out_rows.append(row)
    out = CAND_DIR / "review_v1.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[review] 全量审校文件 {len(out_rows)} 条 → {out}")


def build_round2(by_doc, ids: list[str]) -> None:
    prior: dict[str, dict] = {}
    if ANNO_PATH.is_file():
        for line in ANNO_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                a = json.loads(line)
                prior[a.get("id", "")] = a
    items: dict[str, dict] = {}
    for suite_file, suite in (("candidates_v1.jsonl", "quality"), ("safety_v1_candidates.jsonl", "safety")):
        src = CAND_DIR / suite_file
        if src.is_file():
            for line in src.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    it = json.loads(line)
                    if it["id"] in ids:
                        items[it["id"]] = base_row(it, suite)

    out_rows = []
    for i in ids:
        it = items.get(i)
        if not it:
            print(f"  ! 未找到 {i}")
            continue
        stored = (it.get("authoring") or {}).get("source_excerpt") or ""
        docs = it["supporting_docs"]
        if stored and docs:
            parts = stored.split("\n\n")
            p1 = recover_prefix(by_doc, docs, parts[0]) or parts[0]
            if len(docs) > 1:
                p2, _s = best_match(by_doc, docs[1], (it["question"] or "") + (it.get("ground_truth") or ""))
                excerpt = f"【来源 A·{docs[0]}】\n{p1}\n\n【来源 B·{docs[1]}】\n{p2}"
            else:
                excerpt = p1
        elif docs:
            excerpt = best_match(by_doc, docs[0], (it["question"] or "") + (it.get("ground_truth") or ""))[0]
        else:
            excerpt = "（无来源）"
        a = prior.get(i, {})
        it["source_excerpt"] = excerpt
        it["match_score"] = None
        it["prior_decision"] = a.get("decision") or "未判定"
        it["prior_reason"] = a.get("reason") or ""
        it["prior_checks"] = a.get("checks") or {}
        out_rows.append(it)

    out = CAND_DIR / "review_round2.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[review] 第二轮精审文件 {len(out_rows)} 条 → {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", default="", help="第二轮精审的题号（逗号分隔）")
    args = parser.parse_args()
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    by_doc = load_parents_by_doc()
    if args.ids:
        build_round2(by_doc, [x.strip() for x in args.ids.split(",") if x.strip()])
    else:
        build_full(by_doc)


if __name__ == "__main__":
    main()
