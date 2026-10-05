"""分块器测试：不变量 + 与旧栈对齐的语义（对照结果见 DR-001）。"""
from __future__ import annotations

from ragqa.config.loader import RetrievalConfig
from ragqa.ingestion.chunker import RecursiveTextSplitter, build_parent_child
from ragqa.types import Document

SEPS = ["\n\n", "\n", "。", ".", "！", "？", "!", "?", " ", ""]


def _splitter(size=400, overlap=40):
    return RecursiveTextSplitter(chunk_size=size, chunk_overlap=overlap, separators=SEPS)


def test_deterministic():
    text = "第一句。第二句。第三句。" * 50
    a = _splitter().split_text(text)
    b = _splitter().split_text(text)
    assert a == b and len(a) > 1


def test_size_invariant():
    text = "。".join(f"这是第{i}句测试文本，用于验证分块不变量" for i in range(200))
    chunks = _splitter(400, 40).split_text(text)
    assert chunks and all(len(c) <= 400 for c in chunks)


def test_overlap_between_chunks():
    text = "".join(f"句子{i}。" for i in range(300))
    chunks = _splitter(200, 50).split_text(text)
    assert len(chunks) > 2
    # 相邻块应存在重叠字符（中间层递归后仍由 merge 保证）
    overlaps = sum(1 for a, b in zip(chunks, chunks[1:]) if b[:10] in a)
    assert overlaps >= len(chunks) - 2


def test_chinese_separator_priority():
    text = "甲" * 500 + "。" + "乙" * 300
    chunks = _splitter(400, 40).split_text(text)
    # 已验证与 langchain 原版输出完全一致（含此边界用例）：
    # - 无分隔符的超长段按字递归：400 + 140（含 40 字重叠余量）
    # - keep_separator=True：句号并入下一块开头
    assert chunks[0] == "甲" * 400
    assert chunks[1] == "甲" * 140
    assert chunks[2] == "。" + "乙" * 300


def test_empty_and_short():
    s = _splitter()
    assert s.split_text("") == []
    assert s.split_text("短文本") == ["短文本"]


def test_whitespace_stripped():
    s = _splitter()
    assert s.split_text("\n\n  内容  \n\n") == ["内容"]


def test_build_parent_child_deterministic_ids():
    pages = [Document(page_content="父子分块测试。" * 200, metadata={"doc_id": "d1", "source": "d1.txt", "page": 0})]
    cfg = RetrievalConfig(parent_chunk_size=1200, parent_chunk_overlap=150, child_chunk_size=400, child_chunk_overlap=40)
    parents1, children1 = build_parent_child(pages, cfg)
    parents2, children2 = build_parent_child(pages, cfg)

    assert [p.metadata["parent_id"] for p in parents1] == [p.metadata["parent_id"] for p in parents2]
    assert [c.metadata["chunk_uid"] for c in children1] == [c.metadata["chunk_uid"] for c in children2]
    # 子块回指父块；metadata 传递完整
    pids = {p.metadata["parent_id"] for p in parents1}
    assert all(c.metadata["parent_id"] in pids for c in children1)
    assert all(c.metadata["doc_id"] == "d1" and c.metadata["source"] == "d1.txt" for c in children1)
