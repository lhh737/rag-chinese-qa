"""中文分词工具（jieba），BM25 与查询共用。"""
from __future__ import annotations

import jieba


def zh_tokenize(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    try:
        return [t for t in jieba.cut(text) if t.strip()]
    except Exception:  # jieba 异常时退化为字级切分
        return list(text)
