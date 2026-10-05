"""Run Registry v2（设计 §4.9）：每次运行登记一条，任何报告数字可回溯到完整配置。

JSONL 零依赖（文档注明 SQLite/PG 迁移路径）；run_id 形如 20261005-001（按日递增）。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ragqa.utils.paths import REPO_ROOT

DEFAULT_PATH = REPO_ROOT / "eval_results" / "registry.jsonl"


class RunRegistry:
    def __init__(self, path: str | Path = DEFAULT_PATH) -> None:
        self.path = Path(path)

    # ── 读取 ─────────────────────────────────────────────

    def load(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def next_run_id(self, now: datetime | None = None) -> str:
        day = (now or datetime.now()).strftime("%Y%m%d")
        seq = 1
        for row in self.load():
            rid = str(row.get("run_id", ""))
            if rid.startswith(day + "-"):
                try:
                    seq = max(seq, int(rid.split("-")[1]) + 1)
                except (IndexError, ValueError):
                    continue
        return f"{day}-{seq:03d}"

    # ── 写入 ─────────────────────────────────────────────

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def latest(self) -> dict[str, Any] | None:
        rows = self.load()
        return rows[-1] if rows else None


def build_run_record(
    *,
    run_id: str,
    run_type: str,
    change_type: str,
    change_description: str,
    dataset: dict,
    corpus: dict,
    index: dict,
    retrieval_config: dict,
    generation: dict,
    judge: dict | None = None,
    metrics: dict | None = None,
    coverage: dict | None = None,
    cost: dict | None = None,
    latency: dict | None = None,
    notes: str = "",
    conclusion: str = "baseline",
) -> dict[str, Any]:
    """按 Registry v2 结构组装一条运行记录（字段缺失处以空对象占位）。"""
    from ragqa.trace import git_commit

    return {
        "run_id": run_id,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_type": run_type,                # baseline | ablation | rrf_scan | iteration | defense | acceptance | smoke
        "change_type": change_type,          # baseline | prompt_update | retrieval_config | dataset_update | defense_update
        "change_description": change_description,
        "code_commit": git_commit(),
        "dataset": dataset,                  # {version, hash, n, split}
        "corpus": corpus,                    # {version, hash}
        "index": index,                      # {version, hash, embed_model}
        "retrieval_config": retrieval_config,
        "generation": generation,            # {model, prompt_version, temperature}
        "judge": judge or {},                # {model, prompt_versions:{...}, temperature}
        "metrics": metrics or {},            # {name: {value, ci95:[lo,hi]}}
        "coverage": coverage or {},          # {judge_coverage, judge_errors}
        "cost": cost or {},                  # {total_cny, by_stage:{...}, cache_hit_rate}
        "latency": latency or {},            # {p50_ms, p95_ms, wall_min}
        "notes": notes,
        "conclusion": conclusion,            # adopted | rejected | baseline
    }
