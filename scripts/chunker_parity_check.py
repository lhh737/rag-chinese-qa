"""分块器对照证据脚本：自研 RecursiveTextSplitter vs langchain 原版逐块比对。

运行环境：需要全局 Python（装有 langchain-text-splitters + pypdf），
即依赖迁移前的环境（本仓库 commit 0fb9fd8 的 requirements.txt）：

    PYTHONUTF8=1 python scripts/chunker_parity_check.py

结果作为 DR-001 的证据（2026-10-05 实测：417/417 父块 + 1141/1141 子块 100% 一致）。
"""
from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from pypdf import PdfReader  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter as LC  # noqa: E402

from ragqa.ingestion.chunker import RecursiveTextSplitter as Mine  # noqa: E402

SEPS = ["\n\n", "\n", "。", ".", "！", "？", "!", "?", " ", ""]
FILES = [
    "data/papers/1706.03762.pdf",
    "data/papers/2312.10997.pdf",
    "data/papers/2005.14165.pdf",
    "data/llm_intro.txt",
    "data/test_transformer.txt",
]


def doc_pages(rel: str):
    path = os.path.join(REPO_ROOT, rel)
    if rel.endswith(".pdf"):
        reader = PdfReader(path)
        return [p.extract_text() or "" for p in reader.pages]
    return [open(path, encoding="utf-8").read()]


def main() -> int:
    evidence = {"generated": "2026-10-05", "configs": [], "identical": True}
    for size, overlap in [(1200, 150), (400, 40)]:
        lc = LC(chunk_size=size, chunk_overlap=overlap, separators=SEPS)
        mine = Mine(chunk_size=size, chunk_overlap=overlap, separators=SEPS)
        total = matched = 0
        for rel in FILES:
            for page in doc_pages(rel):
                if not page.strip():
                    continue
                a = lc.split_text(page)
                b = mine.split_text(page)
                total += max(len(a), len(b))
                matched += sum(1 for x, y in zip(a, b) if x == y)
                if a != b:
                    evidence["identical"] = False
        cfg = {"chunk_size": size, "chunk_overlap": overlap, "chunks": total, "matched": matched,
               "rate": round(matched / max(total, 1), 5)}
        evidence["configs"].append(cfg)
        print(f"size={size} overlap={overlap}: {matched}/{total} = {cfg['rate'] * 100:.3f}%")

    out = os.path.join(REPO_ROOT, "eval_results", "parity", "chunker_compare.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=1)
    print(f"evidence -> {out}")
    return 0 if evidence["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
