"""评测框架测试：Registry / 统计 / Judge 缓存 / runner（离线假管道：续跑 + 护栏 + 汇总）。"""
from __future__ import annotations

import json
from datetime import datetime

from ragqa.evaluation.datasets import EvalItem, dump_jsonl
from ragqa.evaluation.judges.cache import JudgeCache
from ragqa.evaluation.metrics import mcnemar, percentile, proportion, wilson_ci
from ragqa.evaluation.registry import RunRegistry
from ragqa.evaluation.runner import (
    EvaluationRunner,
    RunConfig,
    mrr,
    recall_at_k,
    refusal_detected,
)
from ragqa.trace import build_trace
from ragqa.types import QueryResult


class FakePipeline:
    def __init__(self, cost: float = 0.01, answer: str = "根据资料，答案是 42。") -> None:
        self.cost = cost
        self.answer = answer
        self.calls = 0

    def query(self, q: str) -> QueryResult:
        self.calls += 1
        trace = build_trace(question=q, config_id="test-cfg", latency_total_ms=5.0)
        trace["rrf"] = {"strategy": "rrf", "k": 60, "fused": [{"chunk_uid": "1706.03762:c00000", "rrf_score": 0.03}]}
        trace["context"] = {"parent_ids": ["1706.03762:p0000"], "chars": 10, "texts": ["ctx"], "sources": ["s.pdf"]}
        trace["totals"]["cost_total_cny"] = self.cost
        return QueryResult(answer=self.answer, contexts=[], citations=[], trace=trace)


def _mk_dataset(path, n: int = 4):
    items = [
        EvalItem(id=f"q{i+1:03d}", suite="quality", split="dev", type="factual",
                 question=f"问题{i+1}", ground_truth="42", key_claims=["要点"],
                 supporting_docs=["1706.03762"])
        for i in range(n)
    ]
    dump_jsonl(items, path)
    return path


def test_registry_ids_and_append(tmp_path):
    reg = RunRegistry(tmp_path / "registry.jsonl")
    r1 = reg.next_run_id(datetime(2026, 10, 5))
    assert r1 == "20261005-001"
    reg.append({"run_id": r1})
    assert reg.next_run_id(datetime(2026, 10, 5)) == "20261005-002"
    assert reg.latest()["run_id"] == r1


def test_metrics():
    lo, hi = wilson_ci(8, 10)
    assert 0.40 < lo < 0.55 and 0.85 < hi < 0.99
    p = proportion(1, 4)
    assert p["value"] == 0.25 and len(p["ci95"]) == 2
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert mcnemar(0, 0)["p"] == 1.0
    assert mcnemar(10, 1)["p"] < 0.05


def test_judge_cache(tmp_path):
    c = JudgeCache(tmp_path / "cache.db")
    k = JudgeCache.make_key("q1", "答案", "faithfulness", "v1", "m", {"temperature": 0})
    assert c.get(k) is None
    c.set(k, {"verdict": "pass", "evidence": "e", "reason": "r"})
    assert c.get(k)["verdict"] == "pass"
    assert JudgeCache.make_key("q1", "答案2", "faithfulness", "v1", "m") != k
    assert JudgeCache.make_key("q1", "答案", "faithfulness", "v2", "m") != k


def test_rule_helpers():
    trace = {"rrf": {"fused": [{"chunk_uid": "a:c1"}, {"chunk_uid": "b:c1"}, {"chunk_uid": "a:c2"}]}}
    assert recall_at_k(trace, ["a"], 5) == 1.0
    assert recall_at_k(trace, ["b", "c"], 5) == 0.5
    assert mrr(trace, ["b"]) == 0.5
    assert mrr(trace, ["c"]) == 0.0
    assert refusal_detected("根据提供的资料无法回答。") is True
    assert refusal_detected("答案是 42。") is False


def test_runner_resume_and_summary(tmp_path):
    ds = _mk_dataset(tmp_path / "ds.jsonl", n=4)
    reg = RunRegistry(tmp_path / "registry.jsonl")
    pipe = FakePipeline()
    cfg = RunConfig(dataset_path=ds, suite="quality", split="dev", concurrency=2,
                    max_cost=None, run_id="test-001", out_root=tmp_path / "eval")
    runner = EvaluationRunner(pipe, cfg, registry=reg)
    s1 = runner.run()
    assert s1["n_rows"] == 4 and pipe.calls == 4
    assert s1["metrics"]["recall@5_mean"]["value"] == 1.0
    assert s1["metrics"]["false_refusal_rate"]["value"] == 0.0
    assert (tmp_path / "eval" / "test-001" / "report.md").is_file()
    assert reg.latest()["run_id"] == "test-001"
    # 断点续跑：全部已完成 → 不再调用被测系统
    s2 = runner.run()
    assert pipe.calls == 4 and s2["n_rows"] == 4


def test_runner_cost_guard(tmp_path):
    ds = _mk_dataset(tmp_path / "ds.jsonl", n=4)
    pipe = FakePipeline(cost=0.01)
    cfg = RunConfig(dataset_path=ds, suite="quality", concurrency=1, max_cost=0.005,
                    run_id="test-002", out_root=tmp_path / "eval")
    s = EvaluationRunner(pipe, cfg, registry=RunRegistry(tmp_path / "r.jsonl")).run()
    assert s["aborted"] is True and s["n_rows"] < 4


def test_runner_judged_mode(tmp_path):
    ds = _mk_dataset(tmp_path / "ds.jsonl", n=2)

    class FakeJudge:
        model = "fake-judge"

        def judge(self, **kw):
            return {"verdict": "pass", "evidence": "e", "reason": "r",
                    "criterion": kw["criterion"], "prompt_version": kw["prompt_version"]}

    cfg = RunConfig(dataset_path=ds, suite="quality", mode="judged", concurrency=2,
                    max_cost=None, run_id="test-003", out_root=tmp_path / "eval")
    s = EvaluationRunner(FakePipeline(), cfg, judge_client=FakeJudge(),
                         registry=RunRegistry(tmp_path / "r.jsonl")).run()
    assert s["coverage"]["judge_valid"] == 6  # 2 题 × 3 criteria（answer 类）
    results = (tmp_path / "eval" / "test-003" / "results.jsonl").read_text(encoding="utf-8").strip().split("\n")
    rows = [json.loads(l) for l in results]
    assert rows[0]["judge"]["faithfulness"]["verdict"] == "pass"
