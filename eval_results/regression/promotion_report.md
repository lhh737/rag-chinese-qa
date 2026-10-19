# badcase 回流报告（dry-run）

- 生成时间：2026-10-19 12:44:46
- 输入：`eval_results/20261013-003/attribution.jsonl`（60 行）
- 已有集：`data/golden/golden_v1.jsonl` + `data/golden/safety_v1.jsonl`（100 题）
- 阈值：SimHash ≤3 判近重复；5-gram 重合 >0.7 判泄漏。
- **本脚本不改 `data/golden/`；入集必须人工审批 + CHANGELOG + hash。**

## 1. 回归标签（已有失败题，不重复入集）

共 19 条；失败类分布：

| 失败类 | n |
|---|---|
| retrieval_miss.ranked_below_cutoff | 4 |
| retrieval_miss.not_recalled | 4 |
| omission | 4 |
| generation_synthesis_failure | 4 |
| citation_error | 3 |

明细：`eval_results/regression/regression_v1.jsonl`

## 2. 新题候选（三道门）

- 接收：**0** 条（待人工审校，未入集）
- 拒绝：0 条

（无）

明细：`eval_results/regression/promoted_v1.jsonl`

## 3. 现有配额（供下一版决策）

| 题型 | 现有题数 |
|---|---|
| adversarial | 15 |
| colloquial | 10 |
| factual | 30 |
| multi_hop | 15 |
| out_of_scope | 10 |

## 4. 下一步

1. 对 `promoted_v1.jsonl` 做人工审校（题合理 / 答案可溯源 / 支撑文档正确）；
2. 审校通过后，按 `data/golden/CHANGELOG.md` 的格式记版本 + hash；
3. 重新跑健康度检查和基线，确认新题不泄漏、不破坏配额；
4. 回归标签可直接接入 CI 的回归集（不改 golden 版本）。

## 5. 诚实边界

- 本次输入是**已有失败归因**，不是生产流量；因此新增题通常为 0，产出的是回归标签。
- 新题通路的去重/防泄漏/配额门已实现；有真实新 badcase 时，把 `--input` 换成新数据即可。
- 不要把“回归标签”说成“新评测集”；两者在版本管理上是不同资产。
