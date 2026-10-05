"""构建 P2 人工审校文件：候选题 + 来源摘录（离线最佳匹配，零 API）。

背景：候选题的生成出处只记录了 source_doc；审校（快筛三判断：题合理 / 答案可溯源 /
支撑文档正确）需要看到**来源段落原文**。本脚本对每道质量题在其 supporting_docs 的父块中
做 5-gram 重合度最佳匹配，附上摘录与匹配分；安全题附专用语料说明。

输出：eval_results/candidates/review_v1.jsonl（供 apps/review_viewer.html 模式④）
运行：uv run python scripts/build_review_file.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ragqa.utils.paths import resolve_path  # noqa: E402


def ngrams(text: str, n: int = 5) -> set[str]:
    t = "".join(ch for ch in text if not ch.isspace())
    return {t[i : i + n] for i in range(max(len(t) - n + 1, 0))}


def main() -> None:
    cand_dir = REPO_ROOT / "eval_results" / "candidates"
    parents_path = resolve_path("faiss_db/parent_store.json")
    raw = json.loads(parents_path.read_text(encoding="utf-8"))
    by_doc: dict[str, list[str]] = {}
    for blob in raw.values():
        meta = blob.get("metadata") or {}
        did = meta.get("doc_id") or ""
        if did:
            by_doc.setdefault(did, []).append(blob.get("page_content") or "")

    out_rows: list[dict] = []

    for suite_file, suite in (("candidates_v1.jsonl", "quality"), ("safety_v1_candidates.jsonl", "safety")):
        src = cand_dir / suite_file
        if not src.is_file():
            continue
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            row = {
                "id": item["id"], "suite": suite, "type": item.get("type") or item.get("safety_type"),
                "split": item.get("split"), "question": item["question"],
                "ground_truth": item.get("ground_truth"), "key_claims": item.get("key_claims") or [],
                "supporting_docs": item.get("supporting_docs") or [],
                "expected_behavior": item.get("expected_behavior"),
                "authoring": item.get("authoring") or {},
            }
            if suite == "safety":
                st = item.get("safety_type")
                if st == "injection_doc":
                    marker = item.get("injection_marker") or ""
                    row["source_excerpt"] = f"（文档注入题：测试语料见 data/safety_corpus/，marker={marker}）"
                else:
                    row["source_excerpt"] = "（安全题：人工构造，无来源段落）"
                row["match_score"] = None
            else:
                # 优先用生成时留存的摘录；缺失则离线最佳匹配
                excerpt = (item.get("authoring") or {}).get("source_excerpt")
                score = None
                if not excerpt and row["supporting_docs"]:
                    q_grams = ngrams((item.get("question") or "") + (item.get("ground_truth") or ""))
                    best, best_score = "", 0.0
                    for did in row["supporting_docs"]:
                        for p in by_doc.get(did, []):
                            p_grams = ngrams(p)
                            if not p_grams:
                                continue
                            s = len(q_grams & p_grams) / max(len(q_grams), 1)
                            if s > best_score:
                                best, best_score = p, s
                    excerpt, score = best, round(best_score, 3)
                row["source_excerpt"] = (excerpt or "（无来源摘录）")[:1400]
                row["match_score"] = score
            out_rows.append(row)

    out = cand_dir / "review_v1.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[review] 生成 {len(out_rows)} 条 → {out}")


if __name__ == "__main__":
    main()
