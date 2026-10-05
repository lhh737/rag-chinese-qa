# 评测集候选生成 · 自动质检报告（P1）

- 生成时间：2026-10-05；用时 96s；模型 deepseek-flash
- 生成任务 175 → 有效输出 165 → 静态质检通过 156 → LLM 核验通过 155
- LLM 调用 tokens：prompt 173061 / completion 41930，估算成本 ≈ 0.341 元
- 质检阈值（记录在案）：长度 8-200；中文占比 ≥0.25；SimHash 距离 ≤3 判近重复；防泄漏 5-gram 重合度 ≤0.7

## 驳回原因统计（含 QC 余量部分）

- 长度超标: 4
- 完全重复: 4
- SimHash 近重复: 1
- LLM 核验不通过: 1

## 质量候选题配额（已按 split 分配）

| 题型 | 目标 | 实际 | dev | holdout |
|---|---:|---:|---:|---:|
| factual | 30 | 30 | 22 | 8 |
| multi_hop | 15 | 15 | 11 | 4 |
| colloquial | 10 | 10 | 8 | 2 |
| out_of_scope | 10 | 10 | 8 | 2 |
| adversarial | 15 | 15 | 11 | 4 |

## 安全套件（20 题）

| safety_type | 数量 |
|---|---:|
| injection_query | 4（目标 4） |
| injection_doc | 4（目标 4） |
| harmful | 6（目标 6） |
| boundary | 6（目标 6） |

### 文档注入专用语料（data/safety_corpus/，跨块性质已自检）

- safety_doc_plain_1.txt：marker=INJD-D1，跨块变体=False，1 父块 / 2 子块
- safety_doc_plain_2.txt：marker=INJD-D2，跨块变体=False，1 父块 / 2 子块
- safety_doc_split_1.txt：marker=INJD-XC，跨块变体=True，4 父块 / 8 子块
- safety_doc_split_2.txt：marker=INJD-XC，跨块变体=True，4 父块 / 8 子块

## 候选题抽样预览

**[q001·factual·dev]** 论文中哪一项已有工作与作者的研究最相关，并且它处理的是什么问题？

> 参考答案：SuCI 与本文工作最相关，它通过因果解耦来处理人类多模态理解中的主体变化，即主体特定的虚假相关。

**[q002·factual·dev]** 根据原文，评判“生成答案”是否包含“金答案”所要求信息时，如果生成答案冗长但明确包含了核心答案，按哪条规则处理？

> 参考答案：按语义等价规则处理：如果生成答案冗长但明确包含了核心答案，仍应视为满足要求。

**[q003·factual·dev]** 在论文列出的“How Code-LLMs benefit IAs”相关工作中，哪两个方法被归入“Execution”这一子类别，并且同时出现在“Action Grounding”与“Planning”相关工作的列举里？

> 参考答案：Mint 和 ProgPrompt 同时出现在“Action Grounding”与“Planning”相关工作的列举中。
