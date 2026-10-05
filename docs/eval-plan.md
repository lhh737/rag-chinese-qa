# 评测方案（Eval Plan）· rag-chinese-qa

> 版本：v1 · 2026-10-05 · 依据 `TECHNICAL_DESIGN.md v5.0 §4.1` 一页模板
> 配套：`docs/label-reference.md`（标签表全文）· `docs/annotation-guideline.md`（判定规范）

## 1. 目标

量化自研中文 RAG 系统（HyDE + 向量/BM25(RRF) + 重排 + 父子块 + 行内引用）的
**检索 / 生成 / 安全**三维质量，定位失败模式并驱动迭代。

## 2. 决策用途

1. **版本发布门禁**：是否放行（阈值表见设计 §4.10）；
2. **迭代方案采纳**：变更是否显著变好（配对统计判定）；
3. **对外报告**：能力边界与已知限制（README「已知局限」章节数据来源）。

## 3. 范围与假设

- **覆盖**：单轮知识问答；语料 = 106 篇论文/文档（全量索引，32,501 子块）；
- **不覆盖**：多轮对话、工具调用、Agent（扩展路径见 README，接口预留）；
- **假设**：评测执行走统一门面 `RAGPipeline.query`，与线上行为一致（同代码路径）；
- **API 非确定性**：嵌入 document 路径存在 ~1e-2 量级数值波动（DR-001 实测），
  解读 Δ 时以稳定性抽检噪声带为参照。

## 4. 样本设计

| 套件 | 规模 | 结构 | 纪律 |
|---|---|---|---|
| 质量 | 80 | 5 类题型配额（30/15/10/10/15）；**dev 60 / holdout 20** 一次分配 | holdout 仅用于系统验收；日常分析与调参只用 dev；holdout 只报总体指标 |
| 安全 | 20 | injection_query 4 / injection_doc 4（含 2 道跨块拆分）/ harmful 6 / boundary 6 | 全量运行；注入类走**隔离索引**（faiss_db_safety） |

资产：`data/golden/golden_v1.jsonl` + `safety_v1.jsonl`（hash 冻结于 meta；变更走 CHANGELOG）。
构建链：LLM 反向出题 → 自动质检（去重/防泄漏/可溯源核验）→ 人工两轮审校（快筛+精审）→ 冻结。

## 5. 指标与判定

- **设计原则**：代码能测的绝不用 judge；judge 一律二分单一失败模式；**指标 = 通过率**；
  检索用经典 IR 连续指标；拒答双向度量（拒答率 & 误拒率）；比例型默认带 Wilson 区间。
- **检索**：Recall@5（文档级）、MRR（诊断用，无门禁）；
- **生成**：answer_correctness / faithfulness（单题通过率，核心门禁）、citation_accuracy（逐句）；
- **拒答**：oos_refusal_rate、false_refusal_rate（≤5%）；
- **安全**：injection_resistance（硬门禁 =100%）、harmful_refusal、boundary_compliance；
- **运营**：延迟 p50/p95、分阶段成本、judge 覆盖率（错误不计分母、如实披露）。
- 判定边界规则全文：`docs/annotation-guideline.md`。

## 6. 流程与工具

```
裸跑（bare，无 judge）→ 错误分析（读 30-40 条 trace 开放编码）→ judge 设计+校准（TPR/TNR）
→ 正式度量（judged）→ 归因（交叉表+案例库）→ 对照实验迭代 → 安全防御对照
```

工具：`ragqa` CLI（ingest/query/smoke）+ `scripts/run_eval.py`（执行器）+ Run Registry v2
（每次运行完整登记）+ 评审查看器（`apps/review_viewer.html`，读 trace / 盲标 / 抽检 / 审校）+
评测集卫生（holdout 保护、回归池、CHANGELOG）。

## 7. 产出与交付

`eval_results/{run_id}/`：results.jsonl + traces.jsonl + report.md；跨 run 趋势图（P5）；
校准报告（P4）；归因报告与案例库（P5）；决策记录 DR（每轮迭代）；健康度报告（数据集级）。

## 8. 风险与限制

- **统计边界**：n=80 时通过率 95% CI 约 ±9~11pp，**分题型结论为方向性参考**；
- **单标注人**：无第二标注者，以盲标 + 隔日复标（intra-rater）与判定规范补偿；
- **API 非确定性**与 judge 漂移：靠版本冻结 + 稳定性抽检 + 校准复测控制；
- **语料质量上限**：结论可信度 ≤ 语料质量（体检报告与清洗记录留痕，DR-002 为先例）。
