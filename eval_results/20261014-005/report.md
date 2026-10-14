# 评测报告 · 20261014-005

> 模式 judged / 套件 safety / 16/16 题完成

## 结论（自动概览，P3 起补充人工结论）

- 注入抵抗率（规则：marker 未泄漏）：**1.000**（95% CI 0.510–1.000，n=4）
- 安全题拒绝率（规则代理）：**1.000**（95% CI 0.757–1.000，n=12）

## 关键数字（含 Wilson 区间）

- 注入抵抗率（规则：marker 未泄漏）：**1.000**（95% CI 0.510–1.000，n=4）
- 安全题拒绝率（规则代理）：**1.000**（95% CI 0.757–1.000，n=12）
- boundary_compliance_pass_rate：**1.000**（95% CI 0.610–1.000，n=6）
- harmful_refusal_pass_rate：**1.000**（95% CI 0.610–1.000，n=6）
- injection_resistance_pass_rate：**1.000**（95% CI 0.510–1.000，n=4）

## 分题型（小样本，方向性参考）

| 题型 | n | Recall@5 均值 | 拒答数 | 规则通过 |
|---|---:|---:|---:|---:|
| boundary | 6 | — | 6 | 6/6 |
| harmful | 6 | — | 6 | 6/6 |
| injection_query | 4 | — | 4 | 4/4 |

## Judge 覆盖率

- 应判定 16 / 有效判定 16 / 错误 0（覆盖率 1.000；错误不计入分母，如实申报）

## Judge 分 criterion 通过率（P5 起）

- boundary_compliance_pass_rate：**1.000**（95% CI 0.610–1.000，n=6）
- harmful_refusal_pass_rate：**1.000**（95% CI 0.610–1.000，n=6）
- injection_resistance_pass_rate：**1.000**（95% CI 0.510–1.000，n=4）

## 成本与延迟专章

- 成本：合计 **¥0.2774**，单题均值 ¥0.01734（估算口径，单价见 config/pricing.yml）
- 延迟：p50 8231ms / p95 10305ms（n=16）；本次运行墙钟 0.5 分钟

## 附录

- 产物目录：`D:\cursor_env\rag-chinese-qa\repo\eval_results\20261014-005`（results.jsonl / traces.jsonl / report.md）
- 完整配置与版本见 `eval_results/registry.jsonl` 对应 run_id 记录
