# P4b · AI 第二标注分歧裁决

- 生成时间：2026-10-19 17:54:10
- 已裁决：7 / 7

## 汇总

| adjudication | n | 含义 |
|---|---:|---|
| ai_over_strict | 2 | AI 过严，人工 pass 正确 |
| human_missed | 1 | AI fail 正确，原人工 pass 是漏判 |
| unclear | 4 | 规范歧义，需改 guideline |
| 未填 | 0 | — |

## 按 criterion

| criterion | ai_over_strict | human_missed | unclear |
|---|---:|---:|---:|
| answer_correctness | 0 | 0 | 4 |
| citation_accuracy | 1 | 0 | 0 |
| faithfulness | 1 | 1 | 0 |

## 明细

| id | criterion | human | AI | 裁决 | 备注 |
|---|---|---|---|---|---|
| q001 | faithfulness | pass | fail | ai_over_strict | P4规则：整体拒答的“资料不足”整体豁免faithfulness；误拒归ac，人工pass正确 |
| q004 | citation_accuracy | pass | fail | ai_over_strict | 上下文[4]逐字包含Lewis 2020 RAG文献，足以支撑该句；AI读漏参考文献列表 |
| q006 | answer_correctness | pass | fail | unclear | 缺的是补充性解释（K→M饱和/附录A）；guideline v1未写“补充性要点不计缺失” |
| q009 | answer_correctness | pass | fail | unclear | 缺的是背景政策（鼓励公开/[No]可接受），非问题核心条件 |
| q020 | answer_correctness | pass | fail | unclear | 缺的是格式特征解释；核心“平均哪些格式”已答全 |
| q046 | answer_correctness | pass | fail | unclear | 缺的是BYOL-A预训练数据细节；核心“怎么挑事件/用什么模型”已答全 |
| q046 | faithfulness | pass | fail | human_missed | 答案写“移除语音事件”，原文为“非语音事件”，明确矛盾 |

## 读法

- `ai_over_strict` 多 → AI 第二标注可作为“严格预筛”，但会增加人工复核量；
- `human_missed` 多 → 原人工盲标有漏判，需回看校准集；
- `unclear` 多 → 先改判定规范，再改 judge/annotator prompt。
