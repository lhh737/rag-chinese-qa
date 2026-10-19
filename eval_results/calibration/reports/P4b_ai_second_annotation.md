# P4b · AI 第二标注（同族）· 2026-10-19

- 模型：`deepseek-flash`（与 judge/生成同族；**不是独立第二标注人**）
- Prompt：`prompts/annotator/quality_v1.md`（按判定规范，非 judge prompt）
- 样本：50 题（含全部 21 条 fail 题 + 8 条库外 + 通过对照）
- 有效判定：134 条；错误/缺失：0

## 1. 与人工盲标的一致性

| criterion | n | agreement | κ | TPR | TNR | tp/fn/tn/fp |
|---|---:|---:|---:|---:|---:|---|
| answer_correctness | 42 | 0.905 | 0.810 | 1.000 | 0.810 | 21/0/17/4 |
| faithfulness | 42 | 0.952 | 0.481 | 1.000 | 0.951 | 1/0/39/2 |
| citation_accuracy | 42 | 0.976 | 0.656 | 1.000 | 0.976 | 1/0/40/1 |
| oos_refusal | 8 | 1.000 | — | — | 1.000 | 0/0/8/0 |
| **overall** | 134 | 0.948 | 0.836 | 1.000 | 0.937 | 23/0/104/7 |

> TPR = 人工 fail 中 AI 也判 fail；TNR = 人工 pass 中 AI 也判 pass。
> κ 在 fail 样本很少时不稳定；结论以 agreement + 分歧案例为主。

## 2. 分歧案例

共 7 条，明细：`eval_results/calibration/ai_second_disagreements_v1.jsonl`。

本轮 7 条分歧全部为 **AI 判 fail / 人工判 pass**（AI 更严格）；真人裁决可判断是 AI 过严还是人工漏判。

| id | criterion | human | AI | AI 理由 |
|---|---|---|---|---|
| q001 | faithfulness | pass | fail | 整体拒答的“资料不足”仅豁免可核验陈述，但此处为误拒，且答案未包含任何可核验内容，无法支撑参考答案。 |
| q004 | citation_accuracy | pass | fail | 引用[4]指向的上下文块并非直接支撑该句的文献，属跨块张冠李戴。 |
| q006 | answer_correctness | pass | fail | 答案缺少“K 趋近 M 时效用饱和、附录 A 有描述”这一 key_claim。 |
| q009 | answer_correctness | pass | fail | 答案未体现“鼓励公开代码和数据、[No]可接受”这一 key_claim，缺少至少一条关键主张。 |
| q020 | answer_correctness | pass | fail | 答案缺少 key_claims 中关于 Popular 类格式共享扁平、分隔符驱动线性化特征以及 Data 类格式具有显式结构边界的两条陈述。 |
| q046 | answer_correctness | pass | fail | 参考答案缺少 key_claim：BYOL-A 权重来自 AudioSet 和 FSD50K 自监督预训练，未全部体现。 |
| q046 | faithfulness | pass | fail | 答案称“移除语音事件”，但原文移除的是非语音事件，与上下文矛盾。 |

## 3. 诚实边界（必须与数字一起读）

1. **同族**：AI 标注器与 judge、生成模型同为 `deepseek-flash`；这不是独立第二标注人，不能替代真人 κ。
2. **同模型不同 prompt**：只换了判定规范 prompt，模型权重相同；一致性高是预期结果，不能当作强独立性证据。
3. **样本**：50 题、fail 样本集中；κ 的区间很宽，分 criterion 的数字只作方向性参考。
4. **人工裁决待做**：本轮共 7 条分歧，全部为 AI fail / 人工 pass；未复核前，本报告只报 AI–human 一致率，不把分歧直接判成人工错。

## 4. 这能证明什么 / 不能证明什么

- **能**：同一判定规范下，模型能作为第二路预标注通道，减少人工全量标注成本；分歧项可优先送人工。
- **不能**：不能证明人类标注者间信度；不能证明异族模型也会一致；不能替代真人抽检。

## 5. 分歧裁决（7 条，已完成）

人工逐条复核后：

| 裁决 | n | 条目 | 说明 |
|---|---:|---|---|
| `ai_over_strict` | 2 | q001 faithfulness、q004 citation | q001 未应用“整体拒答豁免 faithfulness”；q004 读漏了上下文 [4] 参考文献列表里的 Lewis 2020 RAG 文献 |
| `human_missed` | 1 | q046 faithfulness | 答案写“移除语音事件”，原文是“移除非语音事件”，明确矛盾 |
| `unclear` | 4 | q006/q009/q020/q046 ac | 缺的是补充性/背景要点；guideline v1 未写“补充性/延伸性要点不计缺失” |

**结论**：AI 第二标注能抓到 1 条真实人工漏判，但也会误用规范（q001）、读漏引用块（q004）；4 条 AC 分歧暴露的是 key_claims 粒度问题。**AI 可做预筛，不能替代人工抽检。**

已据此给 `prompts/annotator/quality_v1.md` 补 P4 边界规则（v1.1）：补充性要点不计缺失、整体拒答豁免 faithfulness、引用列表包含所引文献即算支撑。当前 50 条结果用的是 v1；如需证明 v1.1 的改进，建议重跑 50 条对比。

裁决明细：`eval_results/calibration/reports/P4b_ai_disagreements_adjudicated.md`；裁决后 CSV：`eval_results/calibration/ai_second_disagreements_v1_final.csv`。

## 6. 下一步

- 可选：用 v1.1 prompt 重跑 50 条，比较分歧是否减少（不是必须；当前结果已足够支持“AI 预筛 + 人工裁决”的结论）；
- 若有异族模型额度，抽 20 条做异族锚点，比较同族/异族一致性；
- 若目标是“双标注人 κ”，仍需找真人独立盲标 30–50 条。
