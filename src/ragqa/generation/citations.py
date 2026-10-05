"""逐句引用解析：答案句 → [n] 标记 → 被引上下文。

为 citation_accuracy（设计 §4.3）提供代码侧判定基础：
- 标记存在且编号合法（本模块，代码判定）
- 被引陈述是否确被所引上下文支撑（P4 的 judge 判定，不在本模块）
"""
from __future__ import annotations

import re

from ragqa.types import Citation, ContextBlock

_MARKER = re.compile(r"\[(\d{1,3})\]|【(\d{1,3})】")  # 半角 [n] 与全角【n】均接受
_SENT_BOUNDARY = re.compile(r"(?<=[。！？!?；;])")


def split_sentences(text: str) -> list[str]:
    """中文句读切分（按段落 + 句末标点），空句剔除。"""
    out: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        out.extend(s.strip() for s in _SENT_BOUNDARY.split(para) if s.strip())
    return out


def parse_citations(answer: str, contexts: list[ContextBlock]) -> tuple[list[Citation], dict[str, int]]:
    """解析答案中的 [n] 引用并映射到 ContextBlock（rank=n）。

    返回 (逐句引用列表, 统计)：统计用于 trace 与 citation_accuracy 的代码侧口径。
    """
    sentences = split_sentences(answer)
    citations: list[Citation] = []
    n_markers = n_invalid = n_sent_with_marker = 0

    for sent in sentences:
        markers = [int(m.group(1) or m.group(2)) for m in _MARKER.finditer(sent)]
        if markers:
            n_sent_with_marker += 1
        seen: set[int] = set()
        for m in markers:
            n_markers += 1
            if m in seen:  # 同句重复标记只记一次
                continue
            seen.add(m)
            valid = 1 <= m <= len(contexts)
            if not valid:
                n_invalid += 1
            ctx = contexts[m - 1] if valid else None
            citations.append(
                Citation(
                    answer_sentence=sent,
                    marker=m,
                    valid=valid,
                    source_doc=ctx.doc_id if ctx else None,
                    parent_id=ctx.parent_id if ctx else None,
                    supporting_snippet=(ctx.text[:200].replace("\n", " ") if ctx else None),
                )
            )

    stats = {
        "n_sentences": len(sentences),
        "n_sentences_with_marker": n_sent_with_marker,
        "n_markers": n_markers,
        "n_markers_invalid": n_invalid,
    }
    return citations, stats
