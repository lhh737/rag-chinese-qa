# Judge 校准报告 · measurement · 20261013-194910

- 裁判模型：`deepseek-flash` · prompt `v1` · max_tokens 2048
- 标注来源：`blind_labels_merged_v2.jsonl` · 判定 54 条（error 0 / 缓存命中 53 / 未标或跳过 0）
- tokens：prompt 3758 / completion 1531 · 墙钟 0.3 分钟
- 原始判定：`eval_results\calibration\runs\calib_measurement_20261013-194910.jsonl`

> 口径：TPR=人工 fail 中裁判 fail 的比例；TNR=人工 pass 中裁判 pass 的比例；
> κ=Cohen's kappa（小样本 CI 宽，仅参照）；citation_accuracy 主表为 **composed**（语义 ∪ 代码层编号越界）。

## measurement（测量段）

| criterion | n 对 | TPR（Wilson 95%） | TNR（Wilson 95%） | 一致率 | κ | 混淆 tp/fn/tn/fp |
|---|---|---|---|---|---|---|
| answer_correctness | 17 | 1.000 [0.646, 1.000] (7/7) | 1.000 [0.723, 1.000] (10/10) | 1.000 [0.816, 1.000] (17/17) | 1.000 | 7/0/10/0 |
| faithfulness | 17 | 1.000 [0.206, 1.000] (1/1) | 1.000 [0.806, 1.000] (16/16) | 1.000 [0.816, 1.000] (17/17) | 1.000 | 1/0/16/0 |
| citation_accuracy | 17 | 1.000 [0.206, 1.000] (1/1) | 1.000 [0.806, 1.000] (16/16) | 1.000 [0.816, 1.000] (17/17) | 1.000 | 1/0/16/0 |
| oos_refusal | 3 | —（无样本） | 1.000 [0.439, 1.000] (3/3) | 1.000 [0.439, 1.000] (3/3) | — | 0/0/3/0 |

## citation_accuracy：judge 语义层 vs composed（代码层叠加）

| 段 | n 对 | 语义层 一致率 | composed 一致率 | 代码层判负数 |
|---|---|---|---|---|
| measurement | 17 | 1.000 [0.816, 1.000] (17/17) | 1.000 [0.816, 1.000] (17/17) | 0 |

## 分歧清单（人工 vs composed）

- （无分歧）
