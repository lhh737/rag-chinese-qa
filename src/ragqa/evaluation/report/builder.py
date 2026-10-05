"""报告生成（设计 §4.8 结构）。

结构：结论先行（3 条）→ 关键数字（含 CI）→ 分题型（标注方向性）→ 安全/拒答 →
成本与延迟专章 → 附录（配置与版本）。P1 版为自动结论；P3 起由分析阶段补充人工结论。
"""
from __future__ import annotations

from pathlib import Path

_LABEL = {
    "recall@5_mean": "Recall@5（文档级，均值）",
    "mrr_mean": "MRR（均值）",
    "false_refusal_rate": "误拒率（质量题被拒）",
    "answer_answered_rate": "质量题正常作答率",
    "oos_refusal_rate": "库外题正确拒答率",
    "injection_resistance_rule": "注入抵抗率（规则：marker 未泄漏）",
    "safety_refusal_rule": "安全题拒绝率（规则代理）",
}


def _fmt_metric(name: str, m: dict) -> str:
    if m.get("ci95"):
        lo, hi = m["ci95"]
        return f"{_LABEL.get(name, name)}：**{m['value']:.3f}**（95% CI {lo:.3f}–{hi:.3f}，n={m.get('n', m.get('k'))}）"
    return f"{_LABEL.get(name, name)}：**{m['value']:.3f}**（n={m.get('n', '—')}）"


def build_report(summary: dict, out_dir: Path) -> Path:
    m = summary.get("metrics", {})
    lines = [
        f"# 评测报告 · {summary['run_id']}",
        "",
        f"> 模式 {summary.get('mode')} / 套件 {summary.get('suite')} / "
        f"{summary['n_rows']}/{summary['n_items']} 题完成"
        + ("；**⚠ 成本护栏触发提前中止**" if summary.get("aborted") else ""),
        "",
        "## 结论（自动概览，P3 起补充人工结论）",
        "",
    ]
    for name in ("recall@5_mean", "false_refusal_rate", "oos_refusal_rate",
                 "injection_resistance_rule", "safety_refusal_rule"):
        if name in m:
            lines.append(f"- {_fmt_metric(name, m[name])}")
    if not any(name in m for name in ("recall@5_mean", "false_refusal_rate", "oos_refusal_rate",
                                      "injection_resistance_rule", "safety_refusal_rule")):
        lines.append("- （本次运行无可汇总指标）")

    lines += ["", "## 关键数字（含 Wilson 区间）", ""]
    for name, mm in m.items():
        lines.append(f"- {_fmt_metric(name, mm)}")

    lines += ["", "## 分题型（小样本，方向性参考）", "", "| 题型 | n | Recall@5 均值 | 拒答数 | 规则通过 |", "|---|---:|---:|---:|---:|"]
    for t, s in sorted(summary.get("by_type", {}).items()):
        r5 = f"{sum(s['recall@5']) / len(s['recall@5']):.3f}" if s.get("recall@5") else "—"
        mp = f"{s['metric_pass']}/{s['metric_total']}" if s.get("metric_total") else "—"
        lines.append(f"| {t} | {s['n']} | {r5} | {s['refused']} | {mp} |")

    cov = summary.get("coverage") or {}
    if cov:
        lines += ["", "## Judge 覆盖率", "",
                  f"- 应判定 {cov['judge_expected']} / 有效判定 {cov['judge_valid']} / 错误 {cov['judge_errors']}"
                  f"（覆盖率 {cov['judge_coverage']:.3f}；错误不计入分母，如实申报）"]

    cost = summary.get("cost", {})
    lat = summary.get("latency", {})
    lines += [
        "", "## 成本与延迟专章", "",
        f"- 成本：合计 **¥{cost.get('total_cny', 0):.4f}**，单题均值 ¥{cost.get('per_item_cny', 0):.5f}（估算口径，单价见 config/pricing.yml）",
        f"- 延迟：p50 {lat.get('p50_ms', 0):.0f}ms / p95 {lat.get('p95_ms', 0):.0f}ms（n={lat.get('n', 0)}）；"
        f"本次运行墙钟 {lat.get('wall_min', 0)} 分钟",
        "",
        "## 附录",
        "",
        f"- 产物目录：`{summary.get('out_dir', '')}`（results.jsonl / traces.jsonl / report.md）",
        "- 完整配置与版本见 `eval_results/registry.jsonl` 对应 run_id 记录",
    ]
    path = Path(out_dir) / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
