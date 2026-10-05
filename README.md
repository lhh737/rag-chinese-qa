<h1 align="center">RAG-Chinese-QA</h1>
<p align="center">
  中文 RAG 文档问答系统 · <b>以评测体系为主轴的质量工程（升级中）</b><br>
  <b>混合检索(RRF) · HyDE 增强 · 重排序 · 父子块 · 行内引用 [n]</b>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/uv-managed-de5fe9" alt="uv">
</p>
<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>

> 🚧 **升级进行中**：项目正在升级为「以评测体系为主轴」的 RAG 质量工程（六环闭环：构建→评测→可见→归因→迭代→安全）。
> P0 已完成：`src/ragqa` 重构（统一门面 + 结构化 trace + 去 LangChain 依赖 + 全量索引重建）；评测框架与评测集建设中，README 将在收尾阶段全面重构。

---

## 概述

**RAG-Chinese-QA** 是一套面向中文文档的 RAG 端到端系统。上传 PDF / TXT，系统自动分块、向量化并构建索引，随后可用自然语言提问——每个回答附带原文来源引用。

相比单纯的向量检索方案，本系统的核心差异在于 **三级管线**：FAISS + BM25 双路粗筛 → Cross-Encoder 精排 → 父子块上下文还原，在召回率和答案质量之间取得平衡。

**API 模式开箱即用，零模型下载**；也支持切换为离线 BGE 本地模型。

---

## 快速开始

> 前置：Python 3.11+ 与 [uv](https://docs.astral.sh/uv/)；两个 API Key：[DashScope](https://dashscope.aliyuncs.com/)（向量/重排序）+ [DeepSeek](https://platform.deepseek.com/)（生成）。

```bash
git clone https://github.com/lhh737/rag-chinese-qa.git
cd rag-chinese-qa

uv sync                            # 创建 .venv 并安装依赖
cp .env.example .env               # 填入 DASHSCOPE_API_KEY 与 GEN_API_KEY

uv run ragqa ingest                # 构建索引（data/papers 105 篇 + 中文文档，幂等可续跑）
uv run python apps/gradio_ui.py    # 打开 http://127.0.0.1:7860
```

上传文档 → 输入问题 → 获得带 `[n]` 行内引用与来源标注的回答。

---

## 检索管线

```
用户提问
  │
  ├─→ HyDE         LLM 生成假设文档取代原始 query，缩小语义鸿沟
  │
  ├─→ 粗筛          FAISS 向量 (×24)  ─┐
  │                 BM25  稀疏 (×24)  ─┤  RRF 融合（k 可配）→ 候选池 ~48 条
  │                                    ┘
  ├─→ 精排          Cross-Encoder → 重打分 → Top 12
  │
  ├─→ 还原          子块命中 → 向上关联父块 → 动态取 4–8 块
  │
  └─→ 生成          DeepSeek + 上下文 → 回答 + [n] 行内引用
```

---

## 功能详解

### 混合检索

单用向量检索容易遗漏精确术语匹配，单用关键词检索捕捉不到语义同义表达。两路并行召回后用 **RRF（Reciprocal Rank Fusion）** 融合：

| 路径 | 实现 | 优势 |
|------|------|------|
| 向量 (Dense) | DashScope `qwen3.7-text-embedding-flash` / BGE-M3 + faiss-cpu | 同义表达、跨段落语义匹配 |
| 稀疏 (Sparse) | jieba 中文分词 + BM25Okapi 倒排索引 | 专有名词、型号、术语精准命中 |

### HyDE

用户提问倾向于简短口语（"这个模型为什么效果好"），而知识库文本是正式书面语。HyDE (*Hypothetical Document Embeddings*) 先让 LLM 将问题展开为一段教科书风格的段落，再用这段"假想文档"去检索——检索 query 与知识库的语义空间因此对齐，召回率提升显著。LLM 调用失败自动 fallback 为原始问题。

### 父子分块

小块检索精准但丢失上下文，大块语义完整但噪声高。父子分块将两者解耦：

| 粒度 | 大小 | 职责 |
|------|------|------|
| 子块 | ~400 字符 | FAISS 索引 + BM25 语料，负责精确命中 |
| 父块 | ~1,200 字符 | 子块命中后向上关联，还原完整段落供 LLM 阅读 |

### 重排序

RRF 融合后的候选池（约 48 条）送入重排模型逐对计算 `(query, chunk)` 相关性分数，重排后截断至 Top 12。重排模型同时接收 query 和 chunk 做交叉编码，比双塔 embedding 精细得多。

| 后端 | 模型 | 适用场景 |
|------|------|----------|
| DashScope API | `qwen3.7-text-rerank` | 免部署，按量付费 |
| 本地 | `BAAI/bge-reranker-base` | 离线环境，~1GB GPU 显存 |

### 动态 TopK

送给 LLM 的上下文块数量按需自动调节：长问题 +2 块补充上下文；reranker 高分集中时适当放宽以覆盖细节分支。最终输出 4–8 块，在答案完整性和 token 成本间实时平衡。

### 答案溯源

每个回答末尾自动附加引用列表，标注来源文件名和原文摘录（前 320 字符）。Gradio 界面支持点击文档直接查看全文，方便核实信息准确性。

---

## 使用指南

### 运行模式

`.env` 中两行配置即可在 API 与本地模型间切换：

```ini
# API（默认，零下载）          # 本地（离线，需先运行 download_bge.py）
EMBED_MODE=api                  EMBED_MODE=local
RERANK_MODE=api                 RERANK_MODE=local
```

### 界面入口

| 文件 | 端口 | 定位 |
|------|------|------|
| `apps/gradio_ui.py` | 7860 | 单入口：上传/批量导入 papers/、对话（含 [n] 引用）、文档查看 |

### 命令行工具

```bash
uv run ragqa ingest                     # papers/ → 知识库（幂等，可断点续跑）
uv run ragqa documents                  # 已索引文档清单
uv run ragqa query "什么是 RAG？"        # 单条查询（--trace-out 可落盘 trace）
uv run ragqa smoke                      # 冒烟：小索引 + 5 条查询 + Recall@5
python scripts/fetch_papers.py --topic rag --max-per-topic 20   # arXiv 批量下载
python scripts/download_bge.py          # BGE 本地模型（~3.2 GB，离线模式可选）
```

### 检索调参

编辑 `config/retrieval.yml`：

```yaml
vector_fetch_k: 24       # 向量路召回数
bm25_fetch_k: 24         # BM25 路召回数
fusion: rrf              # 融合策略（rrf | merge）
rrf_k: 60                # RRF 融合参数
rerank_top_n: 12         # 重排序保留数
base_context_k: 4        # 最小上下文块数
max_context_k: 8         # 最大上下文块数
parent_chunk_size: 1200  # 父块大小（字符）
child_chunk_size: 400    # 子块大小（字符）
```

- 追求 **更高召回**：增大 `vector_fetch_k` / `bm25_fetch_k`
- 追求 **更快响应**：减小 `rerank_top_n` 和 `max_context_k`
- 文档多为 **长段落**：增大 `parent_chunk_size` 避免截断

---

## 项目结构

```
├── src/ragqa/
│   ├── core.py                  # RAGPipeline 统一门面（评测 / UI / 服务共用）
│   ├── cli.py                   # 命令行：ingest / documents / query / smoke
│   ├── trace.py                 # Trace Schema v1（JSONL）+ 分阶段成本记账
│   ├── types.py                 # 核心数据类型（Document / ContextBlock / Citation…）
│   ├── config/                  # pydantic-settings + YAML 配置加载
│   ├── ingestion/               # 解析(pypdf) + 文档清洗 + 自研分块器 + 数据管道
│   ├── retrieval/               # faiss 直连 + 双路召回(RRF) + 重排 + HyDE
│   ├── generation/              # prompt 版本注册表 + [n] 引用解析 + 生成(含流式)
│   └── models/                  # 模型工厂：DashScope 嵌入/重排 + DeepSeek 生成
├── apps/gradio_ui.py            # Gradio 单入口（7860）
├── prompts/answer/v1.md         # 生成 prompt（版本注册表）
├── config/
│   ├── retrieval.yml            # 检索超参（RRF k / 融合策略 / TopK / 分块）
│   ├── generation.yml           # 生成参数
│   └── pricing.yml              # 成本估算单价（元/1M tokens）
├── tests/                       # pytest 离线测试（假嵌入，无需 API key）
├── scripts/                     # 论文下载 + 依赖迁移对照实验
├── data/papers/                 # 知识库论文（已内置精选论文）
├── eval_results/                # 运行登记与报告（P1+ 建设中）
├── decisions/                   # 决策记录（DR）
└── pyproject.toml  uv.lock
```

---

## 常见问题

<details>
<summary><b>如何获取 API Key？</b></summary>

需要两个 Key：① [阿里云百炼](https://dashscope.aliyuncs.com/) 控制台「API-KEY 管理」创建（用于 embedding / rerank）；② [DeepSeek 开放平台](https://platform.deepseek.com/) 创建（用于生成）。均为按量付费、新用户有赠送额度。
</details>

<details>
<summary><b>支持哪些文件格式？</b></summary>

PDF 和 TXT。PDF 通过 pypdf 提取文本层；扫描版 PDF（纯图片）需先用 OCR 工具预处理为 TXT。
</details>

<details>
<summary><b>切换嵌入模型后索引报错？</b></summary>

系统自动检测 FAISS 维度是否匹配当前嵌入模型，不匹配时重建索引。如仍有问题，删除 `faiss_db/` 目录后重启即可。
</details>

<details>
<summary><b>大文件上传很慢？</b></summary>

瓶颈在嵌入 API 的 QPS 限制（免费额度）。可调大 `child_chunk_size` 减少分块数以加速索引构建，代价是检索粒度变粗。
</details>

<details>
<summary><b>BM25 中文分词不准怎么办？</b></summary>

系统使用 jieba 分词。可通过 `jieba.add_word("专有名词")` 添加自定义词。在 `hybrid_retriever.py` 的 `_zh_tokenize` 函数中添加调用即可生效。
</details>

---

## 许可证

MIT — 可自由使用、修改和分发。
