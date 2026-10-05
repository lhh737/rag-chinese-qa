"""P2 定稿冻结：data/golden/{golden_v1.jsonl, safety_v1.jsonl} + meta(hash) + CHANGELOG + 健康度报告。

纪律：评测集变更走版本号 + CHANGELOG + hash（设计 §4.2）；holdout 终身有效。
运行：uv run python scripts/freeze_dataset.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ragqa.evaluation.datasets import (  # noqa: E402
    QUALITY_QUOTA,
    SAFETY_QUOTA,
    dataset_hash,
    dump_jsonl,
    load_jsonl,
    validate_dataset,
    write_meta,
)

CAND = REPO_ROOT / "eval_results" / "candidates"
OUT = REPO_ROOT / "data" / "golden"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    quality = load_jsonl(CAND / "candidates_v1.jsonl")
    safety = load_jsonl(CAND / "safety_v1_candidates.jsonl")

    # 安全题：第一轮人工全部通过 → 标记
    for s in safety:
        s.authoring = {**(s.authoring or {}), "human_reviewed": True}
        s.authoring.setdefault("method", "hand_written")

    # 校验（全量）
    pq = validate_dataset(quality, "quality")
    ps = validate_dataset(safety, "safety")
    if pq or ps:
        print("!! 校验未通过，停止冻结：")
        for x in pq + ps:
            print("  -", x)
        sys.exit(1)

    dump_jsonl(quality, OUT / "golden_v1.jsonl")
    dump_jsonl(safety, OUT / "safety_v1.jsonl")
    meta_path = write_meta(OUT / "golden_v1.jsonl", OUT / "safety_v1.jsonl", OUT)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # ── CHANGELOG ──
    changelog = f"""# 评测集 CHANGELOG（资产级变更：golden/safety 与 judge prompt 版本）

## golden_v1 / safety_v1 · {date.today().isoformat()}（初始冻结）
- 构成：质量 80（factual 30 / multi_hop 15 / colloquial 10 / out_of_scope 10 / adversarial 15，
  dev 60 / holdout 20）+ 安全 20（injection_query 4 / injection_doc 4〔含 2 道跨块拆分变体〕/ harmful 6 / boundary 6）
- 来源：LLM 反向出题（语料父块）+ 自动质检（去重/防泄漏/可溯源核验）→ 人工两轮审校（快筛 + 精审）
- hash：golden `{meta['golden']['hash'][:16]}…` / safety `{meta['safety']['hash'][:16]}…`（全量见 meta）
- 审校记录：92 条人工通过；9 条第一轮驳回中 7 条经"来源精确复原"后终审接受（工具缺陷所致）、1 条驳回已替换（机器质检）、
  18+8 条未判定经来源修复后由终审处理（依据见 eval_results/review/ 与 data/golden/golden_v1_health.md）
- 已知限制：单标注人；1 条为机器质检替代（未经人工核验，已标注 human_reviewed=false）
"""
    (OUT / "CHANGELOG.md").write_text(changelog, encoding="utf-8")

    # ── 健康度报告 ──
    diff = Counter(i.difficulty or "(未标)" for i in quality)
    claims = [len(i.key_claims) for i in quality]
    hr = sum(1 for i in quality if (i.authoring or {}).get("human_reviewed"))
    lines = [
        "# 评测集健康度报告 · golden_v1 / safety_v1",
        "",
        f"- 生成日期：{date.today().isoformat()}；hash 见 `golden_v1.meta.json`（变更即改版）",
        f"- 质量套件：{len(quality)} 题；安全套件：{len(safety)} 题",
        f"- 人工核验：质量 {hr}/{len(quality)}（含 1 条机器替代未核验）；安全 {len(safety)}/{len(safety)}",
        "",
        "## 分布",
        "",
        "| 题型 | n | dev | holdout | 平均 key_claims |",
        "|---|---:|---:|---:|---:|",
    ]
    for t, q in QUALITY_QUOTA.items():
        sub = [i for i in quality if i.type == t]
        dev = sum(1 for i in sub if i.split == "dev")
        avg_claims = sum(len(i.key_claims) for i in sub) / len(sub) if sub else 0
        lines.append(f"| {t} | {len(sub)} | {dev} | {len(sub)-dev} | {avg_claims:.1f} |")
    lines += [
        "",
        f"难度分布：{dict(diff)}；关键点（key_claims）总数 {sum(claims)}，题均 {sum(claims)/len(claims):.1f}",
        "",
        "## 自动质检（生成阶段）",
        "",
        "- 质检项：长度/语种、完全重复、SimHash 近重复（距离 ≤3）、防泄漏（5-gram 重合度 ≤0.7）、"
        "LLM 可回答性/可溯源/歧义核验；阈值与通过率详见 `eval_results/candidates/candidates_report.md`",
        "",
        "## 人工审校与处置记录",
        "",
        "- 第一轮（100 题快筛）：73 通过 / 9 驳回 / 18 未判定",
        "- 第二轮（27 题精审）：19 通过 / 8 未定（**全部因审校工具缺失来源 B**，非题目缺陷）",
        "- 终审处置：7 条多跳题经确定性回放精确复原来源后接受；1 条口语题驳回并替换（机器质检）",
        "",
        "## 已知限制（随报告披露）",
        "",
        "- 单标注人（无第二标注者；校准阶段将报告标注者自身一致率 intra-rater）",
        "- n=80 时通过率 95% CI 约 ±9~11pp，**分题型结论为方向性参考**",
        "- 1 条替代题未经人工核验（已按 `authoring.human_reviewed=false` 标注）",
        "- holdout 仅用于系统验收，日常分析与改版均不触碰",
    ]
    (OUT / "golden_v1_health.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[freeze] 冻结完成 → {OUT}")
    print(f"[freeze] golden_v1 {len(quality)} 题 hash={meta['golden']['hash'][:16]}…")
    print(f"[freeze] safety_v1 {len(safety)} 题 hash={meta['safety']['hash'][:16]}…")


if __name__ == "__main__":
    main()
