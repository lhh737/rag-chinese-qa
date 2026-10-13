## golden_v1.1 · 2026-10-13（P4 校准勘误）
- 变更：q014（factual/dev）参考答案与 key_claims 的作者名「Jun Qu」→「Jingang Qu」
- 依据：P4 judge 校准与人工盲标对账发现——该题上下文语料 4 处来源中 3 处写作 "Jingang Qu"（1 处来源自身笔误为 "Jun Qu"）；
  基线答案忠实复制多数来源，原 GT 系出题期转录误
- 影响面：golden hash 更新（5b8bfdd9ff4a099d… → a6230a60244edcde…）；q014 判分以修正后 GT 为准（人工盲标 pass 与答案忠实性一致）；其余 79 题内容不变
- 处置依据：设计 §4.2「评测集变更走版本号 + CHANGELOG + hash」；见 P4 校准报告

# 评测集 CHANGELOG（资产级变更：golden/safety 与 judge prompt 版本）

## judge v1 冻结 · 2026-10-13（P4 校准通过）
- 冻结：`prompts/judge/*/v1.md` ×7（哈希见 `eval_results/calibration/reports/P4_calibration_report.md` §1）
- **v1.1 最终口径**：主裁判 `deepseek-flash`（DeepSeek 官方 API 直连，v4.1；同族自偏好局限已声明）+ 第二裁判 `deepseek-v4-pro`（20% 交叉 11/11+11/11）
- v1.0 异族参照：主裁判 `mimo-v2.6-flash` 曾完成全绿校准（存档，见报告 §5 额度事件实录）
- **校准集 164/164 判定有效、零分歧、judge_error 0**；测量段 TPR/TNR 全 1.000（小样本声明见报告）；迭代纪要/事件实录/纪律披露见报告 §4–§6
- 防漂移：prompt 或模型变更须重跑测量段校准（缓存键含 prompt 内容哈希）
- 隔日复标 10 题完成：intra-rater 原始一致率 26/30 = 0.867（报告 §8）；judge v2 候选边界规则见报告 §4


## golden_v1 / safety_v1 · 2026-10-05（初始冻结）
- 构成：质量 80（factual 30 / multi_hop 15 / colloquial 10 / out_of_scope 10 / adversarial 15，
  dev 60 / holdout 20）+ 安全 20（injection_query 4 / injection_doc 4〔含 2 道跨块拆分变体〕/ harmful 6 / boundary 6）
- 来源：LLM 反向出题（语料父块）+ 自动质检（去重/防泄漏/可溯源核验）→ 人工两轮审校（快筛 + 精审）
- hash：golden `5b8bfdd9ff4a099d…` / safety `c9478e60cfbbf4ba…`（全量见 meta）
- 审校记录：92 条人工通过；9 条第一轮驳回中 7 条经"来源精确复原"后终审接受（工具缺陷所致）、1 条驳回已替换（机器质检）、
  18+8 条未判定经来源修复后由终审处理（依据见 eval_results/review/ 与 data/golden/golden_v1_health.md）
- 已知限制：单标注人；1 条为机器质检替代（未经人工核验，已标注 human_reviewed=false）
