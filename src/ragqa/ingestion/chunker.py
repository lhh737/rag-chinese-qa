"""自研递归分块器（替代 langchain-text-splitters，设计 §5.4）。

算法按 langchain-text-splitters 1.1.x 的
`RecursiveCharacterTextSplitter._split_text` / `_split_text_with_regex` /
`_merge_splits` / `_join_docs` 逐行移植（keep_separator=True,
strip_whitespace=True 的默认语义），以保证依赖迁移的检索对照可信。
对照结果见 decisions/DR-001-deps.md。
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from ragqa.config.loader import RetrievalConfig
from ragqa.types import Document
from ragqa.utils.logging import logger

DEFAULT_SEPARATORS = ["\n\n", "\n", "。", ".", "！", "？", "!", "?", " ", ""]


def _split_text_with_regex(text: str, separator: str, keep_separator: bool) -> list[str]:
    """与 langchain 相同：把分隔符并入其后一块（keep_separator=True 语义）。"""
    if separator:
        if keep_separator:
            splits_ = re.split(f"({separator})", text)
            splits = [splits_[i] + splits_[i + 1] for i in range(1, len(splits_), 2)]
            if len(splits_) % 2 == 0:
                splits += splits_[-1:]
            splits = [splits_[0], *splits]
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s]


class RecursiveTextSplitter:
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        separators: list[str] | None = None,
        keep_separator: bool = True,
        strip_whitespace: bool = True,
    ) -> None:
        if chunk_overlap > chunk_size:
            raise ValueError(f"chunk_overlap({chunk_overlap}) 不能大于 chunk_size({chunk_size})")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators or DEFAULT_SEPARATORS
        self._keep_separator = keep_separator
        self._strip_whitespace = strip_whitespace

    # ── 对外 ─────────────────────────────────────────────

    def split_text(self, text: str) -> list[str]:
        return self._split_text(text, self._separators)

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """逐文档切分；每个新块复制原 metadata（与旧栈 split_documents 一致）。"""
        out: list[Document] = []
        for doc in documents:
            for chunk in self.split_text(doc.page_content):
                out.append(Document(page_content=chunk, metadata=dict(doc.metadata)))
        return out

    # ── langchain 算法逐行移植 ───────────────────────────

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        final_chunks: list[str] = []
        # 选择当前可用的分隔符
        separator = separators[-1]
        new_separators: list[str] = []
        for i, s_ in enumerate(separators):
            separator_ = re.escape(s_)
            if not s_:
                separator = s_
                break
            if re.search(separator_, text):
                separator = s_
                new_separators = separators[i + 1 :]
                break

        splits = _split_text_with_regex(text, re.escape(separator), self._keep_separator)

        # 合并小片、递归切大片
        good_splits: list[str] = []
        merge_separator = "" if self._keep_separator else separator
        for s in splits:
            if len(s) < self._chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    final_chunks.extend(self._merge_splits(good_splits, merge_separator))
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    final_chunks.extend(self._split_text(s, new_separators))
        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits, merge_separator))
        return final_chunks

    def _join_docs(self, docs: list[str], separator: str) -> str | None:
        text = separator.join(docs)
        if self._strip_whitespace:
            text = text.strip()
        return text or None

    def _merge_splits(self, splits: Iterable[str], separator: str) -> list[str]:
        separator_len = len(separator)
        docs: list[str] = []
        current_doc: list[str] = []
        total = 0
        for d in splits:
            len_ = len(d)
            if total + len_ + (separator_len if len(current_doc) > 0 else 0) > self._chunk_size:
                if total > self._chunk_size:
                    logger.warning("产生了超过 chunk_size 的块：%d > %d", total, self._chunk_size)
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # 回退到 overlap 以内，同时保证下一块能放进当前片
                    while total > self._chunk_overlap or (
                        total + len_ + (separator_len if len(current_doc) > 0 else 0) > self._chunk_size
                        and total > 0
                    ):
                        total -= len(current_doc[0]) + (separator_len if len(current_doc) > 1 else 0)
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += len_ + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs


# ── 父子块组装 ─────────────────────────────────────────────

def build_parent_child(pages: list[Document], cfg: RetrievalConfig) -> tuple[list[Document], list[Document]]:
    """父子分块：父块还原上下文，子块做检索。

    parent_id 采用确定性 ID（`{doc_id}:p{idx:04d}`），跨重建稳定、可进 trace 对比。
    """
    parent_splitter = RecursiveTextSplitter(
        chunk_size=cfg.parent_chunk_size,
        chunk_overlap=cfg.parent_chunk_overlap,
        separators=cfg.separators,
    )
    child_splitter = RecursiveTextSplitter(
        chunk_size=cfg.child_chunk_size,
        chunk_overlap=cfg.child_chunk_overlap,
        separators=cfg.separators,
    )

    parents = parent_splitter.split_documents(pages)
    children: list[Document] = []
    for i, p in enumerate(parents):
        doc_id = p.metadata.get("doc_id", "doc")
        pid = f"{doc_id}:p{i:04d}"
        p.metadata["parent_id"] = pid
        for c in child_splitter.split_text(p.page_content):
            children.append(
                Document(
                    page_content=c,
                    metadata={
                        "doc_id": doc_id,
                        "source": p.metadata.get("source", ""),
                        "page": p.metadata.get("page"),
                        "parent_id": pid,
                    },
                )
            )
    # 子块确定性 uid（跨重建稳定）
    for j, c in enumerate(children):
        c.metadata["chunk_uid"] = f"{c.metadata['doc_id']}:c{j:05d}"
    return parents, children
