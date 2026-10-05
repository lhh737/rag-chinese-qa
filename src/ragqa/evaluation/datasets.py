"""评测集：Schema v2 + 配额/校验/哈希（设计 §4.2）。

- 质量套件 80 题：5 类题型配额（factual 30 / multi_hop 15 / colloquial 10 /
  out_of_scope 10 / adversarial 15），split 一次分配（dev 60 / holdout 20）
- 安全套件 20 题：4 类（injection_query 4 / injection_doc 4 / harmful 6 / boundary 6），
  全量运行不设 split（split="all"）
- 任何变更走版本号 + CHANGELOG + hash（meta 文件）
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ── 配额矩阵（设计 §4.2）────────────────────────────────────
QUALITY_QUOTA: dict[str, dict[str, int]] = {
    "factual": {"total": 30, "dev": 22, "holdout": 8},
    "multi_hop": {"total": 15, "dev": 11, "holdout": 4},
    "colloquial": {"total": 10, "dev": 8, "holdout": 2},
    "out_of_scope": {"total": 10, "dev": 8, "holdout": 2},
    "adversarial": {"total": 15, "dev": 11, "holdout": 4},
}
SAFETY_QUOTA: dict[str, int] = {
    "injection_query": 4,
    "injection_doc": 4,
    "harmful": 6,
    "boundary": 6,
}

Difficulty = Literal["easy", "medium", "hard"]


class EvalItem(BaseModel):
    """Schema v2 单条（质量/安全共用）。"""

    model_config = ConfigDict(extra="allow")

    id: str
    suite: Literal["quality", "safety"]
    split: Literal["dev", "holdout", "all"] = "dev"
    type: str | None = None                  # 质量题型；安全题为 None
    difficulty: Difficulty | None = None
    question: str
    ground_truth: str | None = None
    key_claims: list[str] = Field(default_factory=list)
    supporting_docs: list[str] = Field(default_factory=list)  # 文档级 ID（= 文件名 stem）
    expected_behavior: Literal["answer", "refuse", "ignore_injection"] = "answer"
    safety_type: str | None = None
    injection_marker: str | None = None
    authoring: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


# ── 读写 ───────────────────────────────────────────────────

def load_jsonl(path: str | Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(EvalItem(**json.loads(line)))
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"{path}:{line_no} schema 校验失败: {e}") from e
    return items


def dump_jsonl(items: list[EvalItem], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json(exclude_none=True) + "\n")


def dataset_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


# ── 校验 ───────────────────────────────────────────────────

def validate_dataset(items: list[EvalItem], expected_suite: str | None = None) -> list[str]:
    """返回问题清单（空 = 通过）。校验：唯一性 / 配额 / split 分布 / 字段完备性。"""
    problems: list[str] = []
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        dup = sorted({x for x in ids if ids.count(x) > 1})
        problems.append(f"id 重复: {dup}")

    for item in items:
        if expected_suite and item.suite != expected_suite:
            problems.append(f"{item.id}: suite 应为 {expected_suite}，实际 {item.suite}")
        if not item.question.strip():
            problems.append(f"{item.id}: question 为空")
        if item.suite == "quality":
            if item.type not in QUALITY_QUOTA:
                problems.append(f"{item.id}: 未知题型 {item.type}")
            if item.split not in ("dev", "holdout"):
                problems.append(f"{item.id}: 质量题 split 必须为 dev/holdout")
            if item.expected_behavior == "answer" and not item.supporting_docs:
                problems.append(f"{item.id}: answer 类题目缺 supporting_docs")
            if item.expected_behavior == "answer" and (not item.ground_truth or not item.key_claims):
                problems.append(f"{item.id}: answer 类题目缺 ground_truth/key_claims")
            if item.type == "out_of_scope" and item.expected_behavior != "refuse":
                problems.append(f"{item.id}: out_of_scope 题的 expected_behavior 应为 refuse")
        else:
            if item.safety_type not in SAFETY_QUOTA:
                problems.append(f"{item.id}: 未知 safety_type {item.safety_type}")
            if item.split != "all":
                problems.append(f"{item.id}: 安全题 split 应为 all")
            if item.safety_type in ("injection_query", "injection_doc"):
                if item.expected_behavior != "ignore_injection":
                    problems.append(f"{item.id}: 注入题 expected_behavior 应为 ignore_injection")
                if not item.injection_marker:
                    problems.append(f"{item.id}: 注入题缺 injection_marker")
            if item.safety_type in ("harmful", "boundary") and item.expected_behavior != "refuse":
                problems.append(f"{item.id}: {item.safety_type} 题 expected_behavior 应为 refuse")

    # 配额与 split 计数
    quality = [i for i in items if i.suite == "quality"]
    if quality:
        for t, q in QUALITY_QUOTA.items():
            sub = [i for i in quality if i.type == t]
            if sub and len(sub) != q["total"]:
                problems.append(f"配额不符: {t} 期望 {q['total']}，实际 {len(sub)}")
            dev = sum(1 for i in sub if i.split == "dev")
            hold = sum(1 for i in sub if i.split == "holdout")
            if sub and (dev, hold) != (q["dev"], q["holdout"]):
                problems.append(f"split 分布不符: {t} 期望 dev{ q['dev']}/holdout{q['holdout']}，实际 dev{dev}/holdout{hold}")
    safety = [i for i in items if i.suite == "safety"]
    if safety:
        for t, n in SAFETY_QUOTA.items():
            sub = [i for i in safety if i.safety_type == t]
            if sub and len(sub) != n:
                problems.append(f"安全配额不符: {t} 期望 {n}，实际 {len(sub)}")
    return problems


def write_meta(golden_path: str | Path, safety_path: str | Path | None, out_dir: str | Path | None = None) -> Path:
    """生成 golden_v1.meta.json：version / hash / 配额与 split 计数 / 生成日期。"""
    golden_path = Path(golden_path)
    items = load_jsonl(golden_path)
    meta: dict[str, Any] = {
        "version": golden_path.stem,
        "generated": date.today().isoformat(),
        "golden": {
            "file": golden_path.name,
            "hash": dataset_hash(golden_path),
            "n": len(items),
            "by_type": {t: sum(1 for i in items if i.type == t) for t in QUALITY_QUOTA},
            "by_split": {s: sum(1 for i in items if i.split == s) for s in ("dev", "holdout")},
        },
    }
    if safety_path and Path(safety_path).is_file():
        s_items = load_jsonl(safety_path)
        meta["safety"] = {
            "file": Path(safety_path).name,
            "hash": dataset_hash(safety_path),
            "n": len(s_items),
            "by_type": {t: sum(1 for i in s_items if i.safety_type == t) for t in SAFETY_QUOTA},
        }
    out = Path(out_dir) if out_dir else golden_path.parent
    meta_path = out / f"{golden_path.stem}.meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta_path
