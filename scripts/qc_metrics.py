"""离线 QC 指标看板：把现有 JSONL 装进内存 SQLite，执行 SQL 并生成 Markdown。

用法：
    uv run python scripts/qc_metrics.py
    uv run python scripts/qc_metrics.py --sql eval_results/analysis/qc_dashboard.sql \
        --out eval_results/analysis/qc_dashboard.md

设计约束：
- 零新增依赖（stdlib sqlite3），不新跑实验、不改原始产物；
- 只计算现有文件能支持的指标；无法从记录恢复的指标（如标注产能、真实人力成本）不补造；
- SQL 单独落在 `eval_results/analysis/qc_dashboard.sql`，本脚本负责建表与执行。
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SQL = REPO_ROOT / "eval_results" / "analysis" / "qc_dashboard.sql"
DEFAULT_OUT = REPO_ROOT / "eval_results" / "analysis" / "qc_dashboard.md"


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _build_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    # 1) 人工盲标 + 三段式分段
    conn.execute("CREATE TABLE labels (id TEXT, criterion TEXT, label TEXT, segment TEXT)")
    seg_path = REPO_ROOT / "eval_results" / "calibration" / "calibration_segments.json"
    segments = json.loads(seg_path.read_text(encoding="utf-8")).get("segments", {}) if seg_path.exists() else {}
    for r in _rows(REPO_ROOT / "eval_results" / "calibration" / "blind_labels_merged_v2.jsonl"):
        conn.execute(
            "INSERT INTO labels VALUES (?, ?, ?, ?)",
            (r.get("id"), r.get("criterion"), r.get("label"), segments.get(r.get("id"), "未分段")),
        )

    # 2) 隔日复标（intra-rater）
    conn.execute("CREATE TABLE relabel (id TEXT, criterion TEXT, label TEXT)")
    for r in _rows(REPO_ROOT / "eval_results" / "calibration" / "blind_relabel_10_v1_export_20261013.jsonl"):
        conn.execute("INSERT INTO relabel VALUES (?, ?, ?)", (r.get("id"), r.get("criterion"), r.get("label")))

    # 3) 数据集审校（两轮；空 decision 记为 undecided）
    conn.execute("CREATE TABLE review (id TEXT, type TEXT, decision TEXT, round INTEGER)")
    for round_no, name in ((1, "p2_review_annotations_v1.jsonl"), (2, "p2_review_round2_annotations_v1.jsonl")):
        for r in _rows(REPO_ROOT / "eval_results" / "review" / name):
            conn.execute(
                "INSERT INTO review VALUES (?, ?, ?, ?)",
                (r.get("id"), r.get("type"), r.get("decision") or "undecided", round_no),
            )

    # 4) 失败归因（基线 run）
    # `primary` 是 SQLite 关键字，列名用 primary_class
    conn.execute("CREATE TABLE attribution (id TEXT, type TEXT, primary_class TEXT)")
    for r in _rows(REPO_ROOT / "eval_results" / "20261013-003" / "attribution.jsonl"):
        conn.execute("INSERT INTO attribution VALUES (?, ?, ?)", (r.get("id"), r.get("type"), r.get("attribution")))

    # 5) Run Registry：成本 / 延迟 / judge 覆盖率 / 指标
    conn.execute(
        """CREATE TABLE runs (
            run_id TEXT, run_type TEXT, n INTEGER, split TEXT,
            total_cny REAL, per_item_cny REAL, p50_ms REAL, p95_ms REAL,
            judge_coverage REAL, ac REAL, false_refusal REAL, recall5 REAL
        )"""
    )
    for r in _rows(REPO_ROOT / "eval_results" / "registry.jsonl"):
        m = r.get("metrics") or {}
        c = r.get("cost") or {}
        lat = r.get("latency") or {}
        cov = (r.get("coverage") or {}).get("judge_coverage")
        conn.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                r.get("run_id"),
                r.get("run_type"),
                (r.get("dataset") or {}).get("n"),
                (r.get("dataset") or {}).get("split"),
                c.get("total_cny"),
                c.get("per_item_cny"),
                lat.get("p50_ms"),
                lat.get("p95_ms"),
                cov,
                (m.get("answer_correctness_pass_rate") or {}).get("value"),
                (m.get("false_refusal_rate") or {}).get("value"),
                (m.get("recall@5_mean") or {}).get("value"),
            ),
        )

    conn.commit()
    return conn


def _parse_sql(path: Path) -> list[tuple[str, str]]:
    """按 `-- @title:` 分段读取 SQL；忽略其他注释行。"""
    queries: list[tuple[str, str]] = []
    title: str | None = None
    buf: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("-- @title:"):
            if title and buf:
                queries.append((title, "\n".join(buf).strip().rstrip(";")))
            title = line.split(":", 1)[1].strip()
            buf = []
        elif line.strip().startswith("--"):
            continue
        else:
            buf.append(line)
    if title and buf:
        queries.append((title, "\n".join(buf).strip().rstrip(";")))
    return queries


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if abs(value) < 1:
            return f"{value:.4f}"
        return f"{value:,.2f}"
    return str(value)


def _md_table(cursor: sqlite3.Cursor) -> str:
    cols = [d[0] for d in cursor.description or []]
    table = cursor.fetchall()
    if not table:
        return "（无数据）"
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for row in table:
        lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sql", default=str(DEFAULT_SQL))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    sql_path = Path(args.sql)
    out_path = Path(args.out)
    if not sql_path.exists():
        print(f"✗ SQL 文件不存在：{sql_path}")
        return 2

    queries = _parse_sql(sql_path)
    if not queries:
        print("✗ 未解析到查询（检查 -- @title: 标记）")
        return 2

    conn = _build_db()
    sections: list[str] = [
        "# QC 指标看板（离线，SQLite）",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 查询文件：`{sql_path.relative_to(REPO_ROOT).as_posix()}`",
        "- 数据源：`eval_results/calibration/blind_labels_merged_v2.jsonl`、"
        "`blind_relabel_10_v1_export_20261013.jsonl`、`eval_results/review/*.jsonl`、"
        "`eval_results/20261013-003/attribution.jsonl`、`eval_results/registry.jsonl`",
        "- 口径：只读现有产物，不新跑实验；无法从记录恢复的指标（产能、真实人力成本）不补造。",
        "",
    ]

    for title, sql in queries:
        cur = conn.execute(sql)
        sections += [f"## {title}", "", _md_table(cur), ""]

    sections += [
        "## 读法说明",
        "",
        "- **合格率**：盲标 `pass` 占比；`fail` 不是标注错误，而是被测系统在该 criterion 上的失败。",
        "- **单人复标一致性**：隔日复标 10 题 × 3 criterion 的原始一致率；不是 inter-annotator κ。",
        "- **审校结果**：`undecided` 是未判定，不是通过；不要把它并入 pass。",
        "- **成本/延迟**：Registry 的 run 级记录；不含校准/裁判费用（P4/P5/P6 报告另计）。",
        "- **judge 覆盖率**：R1 的 `0.9878` 是 rejudge 回填前的 registry 原值；P5 报告经 "
        "`rejudge_errors.py` 回填后为 164/164。holdout 的 `1.0536` 是 registry 的 "
        "valid/expected 计算口径导致 >1；P6 报告按 56/56=1.000 记。看板保留原值，不覆盖。",
        "- **run_id 口径**：holdout 在 Registry 中的记录是 `20261014-004`（run_type=acceptance），"
        "P6 报告正文写的是 `20261014-007`（撞号后迁目录的编号）；引用时以 Registry + P6 §4 说明为准。",
        "- 新增 AI 第二标注后，本看板可加一行 `AI–human agreement`；在写入前不要把它叫成人类 κ。",
        "",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"✓ QC 看板 → {out_path}")
    print(f"  执行 {len(queries)} 条查询；表：labels / relabel / review / attribution / runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
