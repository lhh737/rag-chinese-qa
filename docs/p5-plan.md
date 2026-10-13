# P5 执行计划书 · 正式评测 + 移除式消融 + 归因

> 2026-10-13 · 前置状态：P4 已完成（judge v1 冻结，v1.1 尺子 164/164 零分歧，见
> `eval_results/calibration/reports/P4_calibration_report.md`）。本计划 = P5 的实际执行依据（设计 §P5 的落地版）。

## 1. 模型与额度（谁干什么）

| 角色 | 模型 | 端点 | 说明 |
|---|---|---|---|
| 生成 / HyDE 改写 | `deepseek-flash` | DeepSeek 官方 | 现有 GEN_* 配置，不动 |
| Embedding | `qwen3.7-text-embedding-flash` | 百炼（按量付费） | **P5 不重建索引**，调用量极小；余额约 ¥4.5，设护栏 |
| Rerank | `qwen3.7-text-rerank` | DashScope | 现有配置 |
| **主裁判** | `deepseek-flash`（v4.1） | DeepSeek 官方直连 | **P4 v1.1 已校准**（异族 mimo 同水平交叉验证过） |
| 第二裁判（20% 抽样） | `deepseek-v4-pro` | DeepSeek 官方 | 交叉一致率随每次运行报告 |
| 备用 | `deepseek-v4-flash` | TokenHub | 免费残额，应急 |
| 异族锚点 | `mimo-v2.6-flash` | TokenHub | 残额（~20 万），留给最终验收（holdout）时做一次异族抽检 |
| 工具对照（附录） | RAGAS / DeepEval，LLM backend = `deepseek-flash` | 官方 | P5 收尾时的附录小节的对照执行 |

> 口径声明：v1.1 主裁判与生成模型同族（单族预案，设计 §4.4），自偏好局限随报告披露；异族全绿校准（v1.0）作为参照锚点存档。

## 2. 运行矩阵（先小后大，逐级确认）

| 序 | 运行 | 配置 | 题量 | 模式 | 备注 |
|---|---|---|---|---|---|
| S0 | judged 冒烟 | quality/dev | 10 | judged | 验证 run_eval judged 接线 + 成本统计；**报数后再继续** |
| R1 | **正式基线** | quality/dev | 60 | judged | run_type=baseline；报告与 P3 bare 对照 |
| A1–A4 | 移除式消融 | dev | 60×4 | judged | ①`-rerank`（纯融合序）②`-hyde`（原 query）③`-bm25`（纯稠密）④`fusion=merge`（旧融合行为对照） |
| K1–K4 | RRF k 扫描 | dev | 60×4 | **bare** | k ∈ {10, 30, 60, 100}，看检索层题型差异；若某 k 显著优于默认 60 → 追加 1 个 judged 验证跑 |
| GA | 自动归因 | R1+A* 的 trace | — | — | taxonomy 规则引擎（信号→类别→证据），产出抽检文件 |
| RV | 归因抽检 | 抽样 20 条 | — | 人工 | **你参与 ~30 分钟**（查看器模式③，确认/纠正） |
| RP | 基线报告 | — | — | — | 指标汇总 + 消融结论（配对检验）+ k 扫描 + 归因 + RAGAS/DeepEval 附录 |
| — | holdout | — | — | — | **P5 不跑**（P6 验收专用） |

## 3. 成本估算（官方单价照你账户实测：in ≈ $0.30/M、out ≈ $1.20/M）

| 项 | 量级 | 估费 |
|---|---|---|
| Judge（主裁判）全 P5 | ~7 runs × 180 calls × ~3.5k ≈ **450 万 tokens** | ≈ $1.5–2（¥11–15） |
| 第二裁判（20% 抽样） | ~80 calls ≈ 30 万 | ≈ ¥1–2 |
| 生成侧（7 runs × 60 题） | ≈ 130–200 万 tokens | ≈ ¥3–5 |
| RAGAS/DeepEval 附录 | 20 题 × 2 工具 | ≈ ¥1–2 |
| **合计** | — | **≈ ¥15–25（落于 15–30 元预算）** |

护栏：每 run 带 `--max-cost`；S0 后报实测单题成本，超预期即停下商量。

## 4. 执行顺序

1. **接线**：`scripts/run_eval.py` 补齐 judged 模式的 judge 客户端（官方端点）+ 报告含 judge 指标与成本 —— **S0 冒烟（10 题）→ 报你数字**
2. **R1 正式基线** judged 全量 → 与 P3 bare 指标对照（Recall/MRR/拒答 + ac/faith/citation/oos_pass_rate）
3. **A1–A4 消融**：每跑完一个即时报关键数字；全部完成后做配对检验（McNemar）+ 移除式贡献排序
4. **K1–K4 扫描**（bare）→ 题型差异结论（必要时追加 1 个 judged）
5. **归因**：写规则引擎（taxonomy v1 + 信号证据）→ 生成抽检文件 → **你抽检 20 条（~30min）** → 一致率入报告
6. **报告 + 附录对照**（RAGAS/DeepEval）→ 提交推送
7. P6 预告（迭代闭环 + 安全防御对照 + holdout 验收）

## 5. 风险与预案

| 风险 | 预案 |
|---|---|
| 官方 API 波动 | 自动重试（P4 已验证）+ TokenHub 残额备用；同一 run 内失败项复跑闭环 |
| 费用超预期 | S0 后核实测；每 run `--max-cost`；破预算即停并商量 |
| embedding 余额 | P5 不重建索引（零新增嵌入需求）；如意外需重建 → 先报告 |
| judge 瞬态错误 | 复跑闭环（P4 实测可 100% 补齐）；残留计入覆盖率如实申报 |
| 单族/单标注人局限 | 随报告披露；holdout 验收保留异族锚点抽检 |

## 6. 需要你参与的节点

1. **本计划书确认**（现在）→ 开工
2. S0 冒烟数字确认（<10 分钟）
3. 归因抽检 20 条（~30 分钟，P5 中段）
4. 报告预览 + 推送
