"""离线语料测算（不调用任何 API）：解析 + 分块全部语料，统计字符/分块数。

用途：① 选择冒烟文档集并预估 embedding token 成本；② 语料体检报告的规模基线。
运行：uv run python scripts/corpus_size.py
产出：eval_results/corpus_sizes.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ragqa.config.loader import load_retrieval_config  # noqa: E402
from ragqa.ingestion.chunker import build_parent_child  # noqa: E402
from ragqa.ingestion.loader import load_file  # noqa: E402

TOKEN_PER_CHAR = 0.269  # Kinfra-Text-Embedding-4b 实测（见 DR 与冒烟报告）


def main() -> None:
    cfg = load_retrieval_config()
    files = sorted((REPO_ROOT / "data" / "papers").glob("*.pdf")) + [
        REPO_ROOT / "data" / "llm_intro.txt",
        REPO_ROOT / "data" / "test_transformer.txt",
    ]
    out = {"token_per_char_assumption": TOKEN_PER_CHAR, "docs": {}, "cumulative": []}
    total_chars = total_children = 0
    for i, fp in enumerate(files, start=1):
        t0 = time.time()
        try:
            pages = load_file(fp)
            _parents, children = build_parent_child(pages, cfg)
            chars = sum(len(c.page_content) for c in children)
            doc_chars = sum(len(p.page_content) for p in pages)
        except Exception as e:  # noqa: BLE001
            out["docs"][fp.stem] = {"error": str(e)}
            print(f"[{i}/{len(files)}] {fp.stem}: FAIL {e}")
            continue
        total_chars += chars
        total_children += len(children)
        out["docs"][fp.stem] = {"chars_pages": doc_chars, "chars_children": chars, "n_children": len(children)}
        out["cumulative"].append({"i": i, "doc_id": fp.stem, "cum_chars": total_chars, "cum_children": total_children,
                                  "cum_tokens_est": round(total_chars * TOKEN_PER_CHAR)})
        print(f"[{i}/{len(files)}] {fp.stem}: {chars:,} 字符 / {len(children)} 块 (累计 {total_chars:,} 字符 ≈ {round(total_chars*TOKEN_PER_CHAR):,} tokens) [{time.time()-t0:.1f}s]")

    out["total"] = {"chars": total_chars, "children": total_children, "tokens_est": round(total_chars * TOKEN_PER_CHAR)}
    (REPO_ROOT / "eval_results" / "corpus_sizes.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n合计：{total_chars:,} 字符 / {total_children:,} 子块 ≈ {round(total_chars*TOKEN_PER_CHAR):,} tokens")
    print(f"→ eval_results/corpus_sizes.json")


if __name__ == "__main__":
    main()
