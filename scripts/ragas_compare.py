"""RAGAS 工具对照（附录级）：P4 校准集 52 题（答案类）跑 RAGAS faithfulness，
与人工盲标 + 自研 judge（P4 v1.1 终版）三方对照。

运行（隔离环境，遵守主环境 langchain 零依赖守则）：
    .ragas-env/Scripts/python.exe scripts/ragas_compare.py [--limit N]

输出：
- eval_results/analysis/ragas_compare.jsonl（逐题分数）
- eval_results/analysis/ragas_compare.md（一致率 + 分歧清单）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

from openai import OpenAI  # noqa: E402
from ragas import EvaluationDataset, RunConfig, evaluate  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics._faithfulness import Faithfulness  # noqa: E402  （传统 API，evaluate 兼容；collections 变体不兼容）

JUDGE_RUNS = ["calib_example_20261013-194230", "calib_tuning_20261013-194850",
              "calib_measurement_20261013-194910"]


def _load_key() -> str:
    """读取裁判 key：环境变量 JUDGE_API_KEY 优先，缺省从仓库 .env 读取（该文件不入库）。"""
    key = os.environ.get("JUDGE_API_KEY", "").strip()
    if key:
        return key
    env_path = REPO / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("JUDGE_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("未找到裁判 key：请设置环境变量 JUDGE_API_KEY 或在仓库 .env 中配置")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="仅前 N 题（调试用）")
    args = ap.parse_args()

    items = [json.loads(l) for l in (REPO / "eval_results/calibration/blind_label_v1.jsonl")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    items = [it for it in items if "faithfulness" in (it.get("criteria") or [])]
    if args.limit:
        items = items[: args.limit]

    human = {}
    for l in (REPO / "eval_results/calibration/blind_labels_merged_v2.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        if r["criterion"] == "faithfulness":
            human[r["id"]] = r.get("label")
    judge = {}
    for rid in JUDGE_RUNS:
        for l in (REPO / f"eval_results/calibration/runs/{rid}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            if r["criterion"] == "faithfulness" and r["judge"].get("verdict") in ("pass", "fail"):
                judge[r["id"]] = r["judge"]["verdict"]

    dataset = EvaluationDataset.from_list([
        {
            "user_input": it["question"],
            "response": it["answer"] or "",
            "retrieved_contexts": [c.get("text", "") for c in (it.get("contexts") or [])],
            "reference": it.get("ground_truth") or "",
        }
        for it in items
    ])

    # 实测（2026-10-14）：deepseek-flash 默认思考会挤掉结构化输出预算导致截断/分数错乱
    # （同一题 thinking 开 0.25 / 关 1.0）——必须关思考 + 充足预算。
    llm = llm_factory("deepseek-flash", client=OpenAI(api_key=_load_key(), base_url="https://api.deepseek.com"),
                      max_tokens=8192, extra_body={"thinking": {"type": "disabled"}})
    metric = Faithfulness(llm=llm)
    result = evaluate(dataset, metrics=[metric], run_config=RunConfig(max_workers=6), show_progress=True)

    df = result.to_pandas()
    out_rows = []
    for it, (_, row) in zip(items, df.iterrows()):
        score = row.get("faithfulness")
        out_rows.append({"id": it["id"], "ragas_faithfulness": None if score is None else float(score),
                         "human": human.get(it["id"]), "judge": judge.get(it["id"])})

    runs_dir = REPO / "eval_results/analysis"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / "ragas_compare.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n", encoding="utf-8")

    # 阈值口径：主口径 score==1.0 → pass；敏感性口径 >=0.8 → pass
    def summ(rule_name, pass_rule):
        n = agree_h = agree_j = 0
        dis_h, dis_j = [], []
        for r in out_rows:
            if r["ragas_faithfulness"] is None:
                continue
            n += 1
            rp = "pass" if pass_rule(r["ragas_faithfulness"]) else "fail"
            if r["human"] in ("pass", "fail"):
                if r["human"] == rp:
                    agree_h += 1
                else:
                    dis_h.append((r["id"], r["human"], rp, round(r["ragas_faithfulness"], 3)))
            if r["judge"] in ("pass", "fail"):
                if r["judge"] == rp:
                    agree_j += 1
                else:
                    dis_j.append((r["id"], r["judge"], rp, round(r["ragas_faithfulness"], 3)))
        return n, agree_h, agree_j, dis_h, dis_j

    lines = ["# RAGAS 对照（faithfulness，附录级）\n",
             f"- 样本：校准集答案类 {len(items)} 题（与人工盲标/自研 judge 同题同答案）",
             f"- RAGAS {__import__('ragas').__version__} · LLM backend: deepseek-flash（官方）· 并发 6\n"]
    for name, rule in (("主口径（score==1.0 → pass）", lambda s: s >= 1.0),
                       ("敏感性（>=0.8 → pass）", lambda s: s >= 0.8)):
        n, ah, aj, dh, dj = summ(name, rule)
        lines.append(f"## {name}\n")
        lines.append(f"- RAGAS vs 人工：{ah}/{n} = {(ah / n) if n else 0:.3f}")
        lines.append(f"- RAGAS vs 自研 judge：{aj}/{n} = {(aj / n) if n else 0:.3f}")
        lines.append(f"- 与人工分歧 {len(dh)} 条：" + (", ".join(f"{i}(人工{a}/RAGAS{b}@{c})" for i, a, b, c in dh) or "无"))
        lines.append(f"- 与 judge 分歧 {len(dj)} 条：" + (", ".join(f"{i}(judge{a}/RAGAS{b}@{c})" for i, a, b, c in dj) or "无"))
        lines.append("")
    (runs_dir / "ragas_compare.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("输出：eval_results/analysis/ragas_compare.{jsonl,md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
