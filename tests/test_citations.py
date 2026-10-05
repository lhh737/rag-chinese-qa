"""逐句引用解析测试。"""
from __future__ import annotations

from ragqa.generation.citations import parse_citations, split_sentences
from ragqa.types import ContextBlock


def _ctx(n: int) -> list[ContextBlock]:
    return [
        ContextBlock(rank=i, parent_id=f"p{i}", doc_id=f"doc{i}", source=f"doc{i}.pdf", text=f"上下文 {i} 的内容")
        for i in range(1, n + 1)
    ]


def test_split_sentences():
    assert split_sentences("第一句。第二句！第三句？") == ["第一句。", "第二句！", "第三句？"]
    assert split_sentences("一。\n二。\n\n三") == ["一。", "二。", "三"]


def test_valid_invalid_markers():
    answer = "Transformer 使用自注意力机制 [1]。GPT-3 有 1750 亿参数 [2]。另说 [9]。"
    citations, stats = parse_citations(answer, _ctx(2))
    assert len(citations) == 3
    assert citations[0].valid and citations[0].source_doc == "doc1"
    assert citations[1].valid and citations[1].source_doc == "doc2"
    assert not citations[2].valid and citations[2].source_doc is None
    assert stats == {"n_sentences": 3, "n_sentences_with_marker": 3, "n_markers": 3, "n_markers_invalid": 1}


def test_multi_marker_one_sentence():
    citations, stats = parse_citations("结论由两处证据支撑 [1][2]。", _ctx(2))
    assert [c.marker for c in citations] == [1, 2]
    assert stats["n_markers"] == 2


def test_duplicate_marker_dedup():
    citations, stats = parse_citations("重复引用 [1] 与 [1]。", _ctx(1))
    assert [c.marker for c in citations] == [1]
    assert stats["n_markers"] == 2  # 标记计数照实统计


def test_no_marker():
    citations, stats = parse_citations("这句话没有任何引用。", _ctx(1))
    assert citations == []
    assert stats["n_sentences_with_marker"] == 0


def test_fullwidth_markers():
    citations, _stats = parse_citations("结论见资料【1】，另见【2】。", _ctx(2))
    assert [c.marker for c in citations] == [1, 2]
    assert all(c.valid for c in citations)


def test_arxiv_style_not_matched():
    # arXiv 编号（含点）不应被误认为引用标记
    citations, stats = parse_citations("该结果来自论文 [1706.03762] 的附录。", _ctx(2))
    assert citations == []
    assert stats["n_markers"] == 0
