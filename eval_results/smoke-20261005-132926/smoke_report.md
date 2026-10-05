# P0 冒烟报告（smoke）

- 时间：2026-10-05 14:12:42
- 索引：22 文档 / 9320 子块 / dim=1024 / 模型=kinfra-text-embedding-0.6b
- 导入耗时：2558s（embedding tokens=827562，估算成本 ≈ 0.0000 元）
- 查询数：5；平均 Recall@5（文档级，融合后 top5）：**0.80**
- 查询成本合计（估算）：0.1169 元

## 逐条结果

| 查询 | 期望文档 | top5 文档 | Recall@5 | 引用统计 | 延迟ms |
|---|---|---|---|---|---|
| Transformer 的注意力机制计算复杂度是多少？ | 1706.03762 | test_transformer, 1706.03762, 2309.01431, 2302.13971, 2310.06825 | 1.00 | {'n_sentences': 3, 'n_sentences_with_marker': 3, 'n_markers': 3, 'n_markers_invalid': 0} | 6522 |
| RAG 是什么？它解决了什么问题？ | 2005.11401 | test_transformer, 2312.10997, 2401.05856, llm_intro, 2309.01431 | 0.00 | {'n_sentences': 10, 'n_sentences_with_marker': 10, 'n_markers': 12, 'n_markers_invalid': 0} | 11448 |
| GPT-3 模型有多少参数？ | 2005.14165 | 2005.14165, 2205.01068, llm_intro, test_transformer, 2309.01431 | 1.00 | {'n_sentences': 3, 'n_sentences_with_marker': 3, 'n_markers': 3, 'n_markers_invalid': 0} | 6663 |
| LLaMA 模型的训练数据规模是多少？ | 2302.13971 | 2302.13971, llm_intro, test_transformer, 2309.01431, 2402.05119 | 1.00 | {'n_sentences': 3, 'n_sentences_with_marker': 3, 'n_markers': 3, 'n_markers_invalid': 0} | 6229 |
| 什么是大语言模型？ | llm_intro | llm_intro, 2401.14295, test_transformer, 2401.00812, 2308.11432 | 1.00 | {'n_sentences': 5, 'n_sentences_with_marker': 5, 'n_markers': 5, 'n_markers_invalid': 0} | 6953 |

## 回答与引用（人工抽查用）

### Transformer 的注意力机制计算复杂度是多少？

根据参考资料，自注意力（Self-Attention）每层的计算复杂度为 O(n²·d)，其中 n 是序列长度，d 是表示维度[1]。受限自注意力（restricted self-attention）的每层复杂度为 O(r·n·d)，其中 r 是受限自注意力中邻域的大小[1]。作为对比，循环层（Recurrent）的每层复杂度为 O(n·d²)，卷积层（Convolutional）的每层复杂度为 O(k·n·d²)，其中 k 是卷积核大小[1]。

### RAG 是什么？它解决了什么问题？

RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合检索和生成的技术[1]。它通过在生成答案前先从知识库中检索相关文档，然后将检索结果作为上下文提供给生成模型[1]。RAG 系统结合了信息检索能力与大语言模型的生成能力，其检索组件负责针对用户查询从数据存储中检索相关信息，生成组件则负责将检索到的信息作为上下文来生成答案[4][5]。

RAG 主要解决大语言模型的知识截止和幻觉问题[1]。大语言模型在处理超出其训练数据的查询或需要最新信息的查询时，会产生“幻觉”[2]；同时大语言模型还存在输出内容可能不正确、且无法直接指导或更新输出内容（除提示工程外）的局限[6]。RAG 通过引用外部知识，有效减少了生成事实性错误内容的问题[2]，并能综合出上下文相关、准确且最新的信息[5]。此外，RAG 系统使所有非结构化信息都可被索引和查询，从而减少开发时间，无需构建知识图谱，且数据整理和清洗工作有限[4]。

RAG 的核心步骤包括文档分块、文本嵌入、向量检索、重排序和答案生成[1]。常见的 RAG 优化技术包括 HyDE、混合检索、Reranker 和父子分块等[1]。

### GPT-3 模型有多少参数？

GPT-3 是一个自回归语言模型，具有 1750 亿参数 [3]。在 GPT-3 系列中，最大规模的模型被称为 GPT-3 175B 或“GPT-3”，其参数量为 175.0B [1]。此外，GPT-3 的参数量比此前任何非稀疏语言模型多 10 倍 [3]。

### LLaMA 模型的训练数据规模是多少？

LLaMA-33B 和 LLaMA-65B 在 1.4T tokens 上训练，较小的模型在 1.0T tokens 上训练[1]。所有模型均以 4M tokens 的批量大小进行训练[1]。此外，LLaMA 模型是在数万亿 tokens 上训练的[3]。

### 什么是大语言模型？

大语言模型（Large Language Model, LLM）是指基于 Transformer 架构的大规模预训练语言模型[2]。Transformer 架构由 Vaswani 等人在 2017 年的论文《Attention Is All You Need》中提出，其核心是自注意力机制（Self-Attention），能够捕捉序列中任意两个位置之间的依赖关系[2]。Transformer 完全基于自注意力机制，摒弃了传统的循环神经网络（RNN）和卷积神经网络（CNN），其核心组件包括多头注意力（Multi-Head Attention）、位置编码（Positional Encoding）和前馈神经网络（FFN）[1]。Transformer 为大语言模型的发展奠定了基础，后续的 GPT、BERT、T5 等模型都基于此架构[1]。此外，大语言模型以预测下一个 token 为目标，在极大尺度上进行预训练，从而在其参数中隐式地压缩了世界知识[4]。
