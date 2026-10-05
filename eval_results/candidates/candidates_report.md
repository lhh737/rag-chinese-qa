# 评测集候选生成 · 自动质检报告（P1）

- 生成时间：2026-10-05；用时 413s；模型 deepseek-flash
- 生成任务 175 → 有效输出 167 → 静态质检通过 157 → LLM 核验通过 157
- LLM 调用 tokens：prompt 172832 / completion 43531，估算成本 ≈ 0.347 元
- 质检阈值（记录在案）：长度 8-200；中文占比 ≥0.25；SimHash 距离 ≤3 判近重复；防泄漏 5-gram 重合度 ≤0.7

## 驳回原因统计（含 QC 余量部分）

- 长度超标: 6
- 完全重复: 2
- SimHash 近重复: 1
- 中文占比过低: 1

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

**[q001·factual·dev]** 在论文提到的多模态人格理解任务中，作者称自己首次尝试设计什么类型的网络来解决主体偏差问题？

> 参考答案：作者首次尝试设计一种定制的因果干预网络（customized causal intervention network），用于解决多模态人格理解中的主体偏差（subject bias）。

**[q002·factual·dev]** 在评估生成答案是否包含金标准答案所需信息时，如果生成答案冗长但明显包含了核心答案，根据给出的规则应如何判定？

> 参考答案：应判定为正确。因为规则规定，只要生成答案正确提供了金标准答案所要求的核心信息，额外的正确或无关信息不影响判断，且语义等价原则允许冗长但明确包含核心答案的情况。

**[q003·factual·dev]** 在论文列出的表格中，属于“Prompting Based Meth.”这一类别的方法有哪些？请列出其中至少两个，并说明它们与“Finetuning Based Meth.”在分类上的区别。

> 参考答案：属于“Prompting Based Meth.”的方法包括 Mint (Wang et al., 2023h) 和 Self-Debugging (Chen et al., 2023c)。它们与“Finetuning Based Meth.”的区别在于：前者是基于提示的方法，后者是基于微调的方法，二者在表格中被归入不同的方法类别。
