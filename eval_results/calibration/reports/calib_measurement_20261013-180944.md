# Judge 校准报告 · measurement · 20261013-180944

- 裁判模型：`mimo-v2.5-pro` · prompt `v1` · max_tokens 2048
- 标注来源：`blind_labels_merged_v2.jsonl` · 判定 54 条（error 3 / 缓存命中 0 / 未标或跳过 0）
- tokens：prompt 172228 / completion 60915 · 墙钟 8.5 分钟
- 原始判定：`eval_results\calibration\runs\calib_measurement_20261013-180944.jsonl`

> 口径：TPR=人工 fail 中裁判 fail 的比例；TNR=人工 pass 中裁判 pass 的比例；
> κ=Cohen's kappa（小样本 CI 宽，仅参照）；citation_accuracy 主表为 **composed**（语义 ∪ 代码层编号越界）。

## measurement（测量段）

| criterion | n 对 | TPR（Wilson 95%） | TNR（Wilson 95%） | 一致率 | κ | 混淆 tp/fn/tn/fp |
|---|---|---|---|---|---|---|
| answer_correctness | 17 | 1.000 [0.646, 1.000] (7/7) | 0.900 [0.596, 0.982] (9/10) | 0.941 [0.730, 0.990] (16/17) | 0.881 | 7/0/9/1 |
| faithfulness | 17 | 0.000 [0.000, 0.793] (0/1) | 1.000 [0.806, 1.000] (16/16) | 0.941 [0.730, 0.990] (16/17) | 0.000 | 0/1/16/0 |
| citation_accuracy | 14 | 0.000 [0.000, 0.793] (0/1) | 1.000 [0.772, 1.000] (13/13) | 0.929 [0.685, 0.987] (13/14) | 0.000 | 0/1/13/0 |
| oos_refusal | 3 | —（无样本） | 1.000 [0.439, 1.000] (3/3) | 1.000 [0.439, 1.000] (3/3) | — | 0/0/3/0 |

## citation_accuracy：judge 语义层 vs composed（代码层叠加）

| 段 | n 对 | 语义层 一致率 | composed 一致率 | 代码层判负数 |
|---|---|---|---|---|
| measurement | 14 | 0.929 [0.685, 0.987] (13/14) | 0.929 [0.685, 0.987] (13/14) | 0 |

## 第二裁判交叉（deepseek-v4-flash，每 5 条抽 1）

- 裁判间一致率：10/10 = 1.000
- 第二裁判 vs 人工一致率：11/11 = 1.000

## 分歧清单（人工 vs composed）

- **q003 · citation_accuracy**：人工 **fail**（系统回答引用 [5]，称“被归入 Execution 子类别且同时出现在 Action Grounding 与 Planning 相关工作中列举的两个方法是 Progprompt 和 Code as Policies”。） → 裁判 **pass**（引用块[5]支撑了陈述中所有主张。）
- **q003 · faithfulness**：人工 **fail**（参考资料中 Planning 的典型例子确实出现 Progprompt 和 Code as Policies，但并不能支持 Code as Policies 同时被归入“Execution”下的“Action Groun） → 裁判 **pass**（所有可核验陈述均被上下文直接支撑。）
- **q006 · answer_correctness**：人工 **pass**（） → 裁判 **fail**（缺失 key_claims 的第三条：当 K 趋近 M 时效用饱和，该现象在附录 A 中描述。）
