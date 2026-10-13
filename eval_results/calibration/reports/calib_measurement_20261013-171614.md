# Judge 校准报告 · measurement · 20261013-171614

- 裁判模型：`minimax-m2.7` · prompt `v1` · max_tokens 2048
- 标注来源：`blind_labels_merged_v2.jsonl` · 判定 54 条（error 3 / 缓存命中 0 / 未标或跳过 0）
- tokens：prompt 136133 / completion 41183 · 墙钟 8.1 分钟
- 原始判定：`eval_results\calibration\runs\calib_measurement_20261013-171614.jsonl`

> 口径：TPR=人工 fail 中裁判 fail 的比例；TNR=人工 pass 中裁判 pass 的比例；
> κ=Cohen's kappa（小样本 CI 宽，仅参照）；citation_accuracy 主表为 **composed**（语义 ∪ 代码层编号越界）。

## measurement（测量段）

| criterion | n 对 | TPR（Wilson 95%） | TNR（Wilson 95%） | 一致率 | κ | 混淆 tp/fn/tn/fp |
|---|---|---|---|---|---|---|
| answer_correctness | 17 | 1.000 [0.646, 1.000] (7/7) | 0.800 [0.490, 0.943] (8/10) | 0.882 [0.657, 0.967] (15/17) | 0.767 | 7/0/8/2 |
| faithfulness | 17 | 1.000 [0.206, 1.000] (1/1) | 1.000 [0.806, 1.000] (16/16) | 1.000 [0.816, 1.000] (17/17) | 1.000 | 1/0/16/0 |
| citation_accuracy | 14 | 0.000 [0.000, 0.793] (0/1) | 1.000 [0.772, 1.000] (13/13) | 0.929 [0.685, 0.987] (13/14) | 0.000 | 0/1/13/0 |
| oos_refusal | 3 | —（无样本） | 1.000 [0.439, 1.000] (3/3) | 1.000 [0.439, 1.000] (3/3) | — | 0/0/3/0 |

## citation_accuracy：judge 语义层 vs composed（代码层叠加）

| 段 | n 对 | 语义层 一致率 | composed 一致率 | 代码层判负数 |
|---|---|---|---|---|
| measurement | 14 | 0.929 [0.685, 0.987] (13/14) | 0.929 [0.685, 0.987] (13/14) | 0 |

## 分歧清单（人工 vs composed）

- **q003 · citation_accuracy**：人工 **fail**（系统回答引用 [5]，称“被归入 Execution 子类别且同时出现在 Action Grounding 与 Planning 相关工作中列举的两个方法是 Progprompt 和 Code as Policies”。） → 裁判 **pass**（引用标注[5]对应的上下文块明确列出 Progprompt 与 Code as Policies 属于 Execution 子类别并在 Action Grounding 与 Planning 中均出现，句子得到支撑且无越）
- **q006 · answer_correctness**：人工 **pass**（） → 裁判 **fail**（缺失关键要点：当 K 趋近 M 时效用会出现饱和、该现象在附录 A 中描述——系统只回答了 M > K 的直接目的，未涵盖 ground_truth 中关于效用饱和现象及其位置说明的要点）
- **q046 · answer_correctness**：人工 **pass**（） → 裁判 **fail**（缺失 key_claims 第4条：BYOL-A 的权重来源（AudioSet 和 FSD50K 上的自监督预训练）未提及；且系统回答将'移除语音事件'表述为'移除语音事件'，而问题问的是非语音事件处理，此处存在语义偏差）
