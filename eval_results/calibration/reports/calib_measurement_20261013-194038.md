# Judge 校准报告 · measurement · 20261013-194038

- 裁判模型：`deepseek-flash` · prompt `v1` · max_tokens 2048
- 标注来源：`blind_labels_merged_v2.jsonl` · 判定 54 条（error 2 / 缓存命中 3 / 未标或跳过 0）
- tokens：prompt 148465 / completion 40408 · 墙钟 2.5 分钟
- 原始判定：`eval_results\calibration\runs\calib_measurement_20261013-194038.jsonl`

> 口径：TPR=人工 fail 中裁判 fail 的比例；TNR=人工 pass 中裁判 pass 的比例；
> κ=Cohen's kappa（小样本 CI 宽，仅参照）；citation_accuracy 主表为 **composed**（语义 ∪ 代码层编号越界）。

## measurement（测量段）

| criterion | n 对 | TPR（Wilson 95%） | TNR（Wilson 95%） | 一致率 | κ | 混淆 tp/fn/tn/fp |
|---|---|---|---|---|---|---|
| answer_correctness | 17 | 1.000 [0.646, 1.000] (7/7) | 0.800 [0.490, 0.943] (8/10) | 0.882 [0.657, 0.967] (15/17) | 0.767 | 7/0/8/2 |
| faithfulness | 17 | 1.000 [0.206, 1.000] (1/1) | 1.000 [0.806, 1.000] (16/16) | 1.000 [0.816, 1.000] (17/17) | 1.000 | 1/0/16/0 |
| citation_accuracy | 15 | 1.000 [0.206, 1.000] (1/1) | 1.000 [0.785, 1.000] (14/14) | 1.000 [0.796, 1.000] (15/15) | 1.000 | 1/0/14/0 |
| oos_refusal | 3 | —（无样本） | 1.000 [0.439, 1.000] (3/3) | 1.000 [0.439, 1.000] (3/3) | — | 0/0/3/0 |

## citation_accuracy：judge 语义层 vs composed（代码层叠加）

| 段 | n 对 | 语义层 一致率 | composed 一致率 | 代码层判负数 |
|---|---|---|---|---|
| measurement | 15 | 1.000 [0.796, 1.000] (15/15) | 1.000 [0.796, 1.000] (15/15) | 0 |

## 第二裁判交叉（deepseek-v4-pro，每 5 条抽 1）

- 裁判间一致率：10/11 = 0.909
- 第二裁判 vs 人工一致率：11/11 = 1.000
- 裁判分歧：**q068 · answer_correctness**：主裁判 fail vs 第二裁判 pass（答案已明确纠正前提错误并给出正确成本与范围，覆盖问题直接索要的要点。）

## 分歧清单（人工 vs composed）

- **q021 · answer_correctness**：人工 **pass**（） → 裁判 **fail**（间接关联的示例路径方向与 ground_truth 相反：应为 BP CAPABLE_OF CellType，而答案写成 BP←CAPABLE_OF CellType（即 CellType CAPABLE_OF BP）。）
- **q068 · answer_correctness**：人工 **pass**（） → 裁判 **fail**（答案虽纠正了上限并给出正确范围，但另称资料存在$77–$87 million表述，与ground_truth中原文范围$7.7–$8.7 million、并非$77–$87 million相抵触。）
