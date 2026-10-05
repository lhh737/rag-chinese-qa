"""评测集基座测试：Schema v2、配额校验、hash/meta、质检工具函数。"""
from __future__ import annotations

import json

from ragqa.evaluation.datasets import (
    EvalItem,
    dataset_hash,
    dump_jsonl,
    load_jsonl,
    validate_dataset,
    write_meta,
)
from ragqa.evaluation.generate import cn_ratio, hamming, leak_ratio, simhash


def _quality_item(i: int, qtype: str = "factual", split: str = "dev") -> EvalItem:
    return EvalItem(
        id=f"q{i:03d}", suite="quality", split=split, type=qtype,
        question="这是一道测试问题？", ground_truth="测试答案", key_claims=["要点一"],
        supporting_docs=["1706.03762"],
    )


def test_roundtrip_and_hash(tmp_path):
    items = [_quality_item(1), _quality_item(2, split="holdout")]
    p = tmp_path / "golden.jsonl"
    dump_jsonl(items, p)
    loaded = load_jsonl(p)
    assert [i.id for i in loaded] == ["q001", "q002"]
    assert dataset_hash(p) == dataset_hash(p) and len(dataset_hash(p)) == 64


def test_validate_ok(tmp_path):
    # 子集只允许出现配额/分布类提示（这是对完整集的约束）
    items = [_quality_item(1), _quality_item(2, split="holdout")]
    problems = validate_dataset(items, "quality")
    assert all(("配额不符" in p or "split 分布不符" in p) for p in problems), problems


def test_validate_catches_duplicates_and_fields():
    bad = [
        _quality_item(1),
        _quality_item(1),  # 重复 id
        EvalItem(id="q003", suite="quality", split="dev", type="factual", question="", ground_truth=""),
    ]
    problems = validate_dataset(bad, "quality")
    assert any("id 重复" in p for p in problems)
    assert any("question 为空" in p for p in problems)


def test_validate_safety_rules():
    bad = EvalItem(id="s001", suite="safety", split="all", question="测试", safety_type="injection_query",
                   expected_behavior="refuse")  # 注入题应为 ignore_injection 且带 marker
    problems = validate_dataset([bad], "safety")
    assert any("ignore_injection" in p for p in problems)
    assert any("injection_marker" in p for p in problems)


def test_write_meta(tmp_path):
    p = tmp_path / "golden_v1.jsonl"
    dump_jsonl([_quality_item(1)], p)
    meta_path = write_meta(p, None)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["golden"]["n"] == 1 and meta["golden"]["hash"]


def test_qc_tools():
    # 中文占比：术语夹英文不应被误杀
    assert cn_ratio("GPT-3 的参数量是多少？") > 0.25
    assert cn_ratio("What is the parameter count?") < 0.05
    # SimHash：完全相同/仅空白差异 → 距离极小；无关文本 → 距离显著更大
    a = "Transformer 的注意力机制计算复杂度是多少？"
    b = "Transformer 的注意力机制复杂度是多少？"
    c = "如何用空气炸锅做蛋糕？"
    assert hamming(simhash(a), simhash(a + " ")) <= 3   # 空白被归一化
    assert hamming(simhash(a), simhash(b)) < hamming(simhash(a), simhash(c))
    # 防泄漏：逐字照抄 → 高重合；改写 → 低重合
    src = "自注意力机制的计算复杂度与序列长度的平方成正比。"
    assert leak_ratio(src, src) > 0.9
    assert leak_ratio("这个机制的代价随长度怎么变？", src) < 0.3
