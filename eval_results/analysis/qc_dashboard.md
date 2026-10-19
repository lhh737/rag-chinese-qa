# QC 指标看板（离线，SQLite）

- 生成时间：2026-10-19 14:46:34
- 查询文件：`eval_results/analysis/qc_dashboard.sql`
- 数据源：`eval_results/calibration/blind_labels_merged_v2.jsonl`、`blind_relabel_10_v1_export_20261013.jsonl`、`eval_results/review/*.jsonl`、`eval_results/20261013-003/attribution.jsonl`、`eval_results/registry.jsonl`
- 口径：只读现有产物，不新跑实验；无法从记录恢复的指标（产能、真实人力成本）不补造。

## 1. 标注合格率（按 criterion）

| criterion | n | pass_n | fail_n | pass_rate |
|---|---|---|---|---|
| answer_correctness | 52 | 31 | 21 | 0.5962 |
| citation_accuracy | 52 | 51 | 1 | 0.9808 |
| faithfulness | 52 | 51 | 1 | 0.9808 |
| oos_refusal | 8 | 8 | 0 | 1.00 |

## 2. 分段合格率（segment × criterion）

| segment | criterion | n | pass_n | pass_rate |
|---|---|---|---|---|
| 测量段 | answer_correctness | 17 | 10 | 0.5882 |
| 测量段 | citation_accuracy | 17 | 16 | 0.9412 |
| 测量段 | faithfulness | 17 | 16 | 0.9412 |
| 测量段 | oos_refusal | 3 | 3 | 1.00 |
| 示例段 | answer_correctness | 18 | 11 | 0.6111 |
| 示例段 | citation_accuracy | 18 | 18 | 1.00 |
| 示例段 | faithfulness | 18 | 18 | 1.00 |
| 示例段 | oos_refusal | 2 | 2 | 1.00 |
| 调参段 | answer_correctness | 17 | 10 | 0.5882 |
| 调参段 | citation_accuracy | 17 | 17 | 1.00 |
| 调参段 | faithfulness | 17 | 17 | 1.00 |
| 调参段 | oos_refusal | 3 | 3 | 1.00 |

## 3. 单人复标一致性（10 题 × 3 criterion）

| compared | same_n | changed_n | raw_agreement |
|---|---|---|---|
| 30 | 26 | 4 | 0.8667 |

## 4. 数据集审校结果（按轮次）

| round | pass_n | reject_n | undecided_n | total_n |
|---|---|---|---|---|
| 1 | 73 | 9 | 18 | 100 |
| 2 | 19 | 0 | 8 | 27 |

## 5. 失败归因分布（排除“无问题”）

| failure_class | n |
|---|---|
| retrieval_miss.ranked_below_cutoff | 4 |
| retrieval_miss.not_recalled | 4 |
| omission | 4 |
| generation_synthesis_failure | 4 |
| citation_error | 3 |

## 6. 运行成本 / 延迟 / judge 覆盖率（同集合决策样本）

| run_id | run_type | n | split | total_cny | per_item_cny | p50_ms | p95_ms | judge_coverage | answer_correctness | false_refusal | recall_at_5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261013-003 | baseline | 60 | dev | 1.21 | 0.0201 | 9,556.00 | 14,364.00 | 0.9878 | — | 0.2308 | 0.7167 |
| 20261013-004 | ablation | 60 | dev | 0.1935 | 0.0032 | 7,532.00 | 13,086.00 | 0.9939 | 0.5686 | 0.3654 | 0.7667 |
| 20261013-005 | ablation | 60 | dev | 0.6739 | 0.0112 | 5,244.00 | 10,614.00 | 0.9695 | 0.7255 | 0.2308 | 0.7583 |
| 20261014-004 | acceptance | 20 | holdout | 0.3360 | 0.0140 | 4,888.00 | 9,210.00 | 1.05 | 0.6667 | 0.1667 | 0.8000 |
| 20261016-002 | agentic | 60 | dev | 1.05 | 0.0176 | 5,531.00 | 9,013.00 | 0.9878 | 0.7647 | 0.2885 | 0.7750 |
| 20261019-001 | baseline | 30 | dev | 0.1975 | 0.0066 | 5,084.00 | 9,412.00 | 1.00 | 0.6923 | 0.2308 | 0.8250 |

## 读法说明

- **合格率**：盲标 `pass` 占比；`fail` 不是标注错误，而是被测系统在该 criterion 上的失败。
- **单人复标一致性**：隔日复标 10 题 × 3 criterion 的原始一致率；不是 inter-annotator κ。
- **审校结果**：`undecided` 是未判定，不是通过；不要把它并入 pass。
- **成本/延迟**：Registry 的 run 级记录；不含校准/裁判费用（P4/P5/P6 报告另计）。
- **judge 覆盖率**：R1 的 `0.9878` 是 rejudge 回填前的 registry 原值；P5 报告经 `rejudge_errors.py` 回填后为 164/164。holdout 的 `1.0536` 是 registry 的 valid/expected 计算口径导致 >1；P6 报告按 56/56=1.000 记。看板保留原值，不覆盖。
- **run_id 口径**：holdout 在 Registry 中的记录是 `20261014-004`（run_type=acceptance），P6 报告正文写的是 `20261014-007`（撞号后迁目录的编号）；引用时以 Registry + P6 §4 说明为准。
- 新增 AI 第二标注后，本看板可加一行 `AI–human agreement`；在写入前不要把它叫成人类 κ。
