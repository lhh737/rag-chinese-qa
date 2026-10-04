<h1 align="center">RAG-Chinese-QA</h1>
<p align="center">
  基于 LangChain 的中文 RAG 文档问答系统<br>
  <b>混合检索 · HyDE 增强 · 重排序 · 答案溯源</b>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/langchain-1.2-orange" alt="LangChain">
</p>
<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>

---

## 概述

**RAG-Chinese-QA** 是一套面向中文文档的 RAG 端到端系统。上传 PDF / TXT，系统自动分块、向量化并构建索引，随后可用自然语言提问——每个回答附带原文来源引用。

相比单纯的向量检索方案，本系统的核心差异在于 **三级管线**：FAISS + BM25 双路粗筛 → Cross-Encoder 精排 → 父子块上下文还原，在召回率和答案质量之间取得平衡。

**API 模式开箱即用，零模型下载**；也支持切换为离线 BGE 本地模型。

---

## 快速开始

> 前置：Python 3.10+，[DashScope API Key](https://dashscope.aliyuncs.com/)（阿里云百炼，注册即送免费额度）。

```bash
git clone https://github.com/lhh737/rag-chinese-qa.git
cd rag-chinese-qa

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # 编辑 .env，填入 DASHSCOPE_API_KEY
python mvp_app.py                  # 打开 http://127.0.0.1:7860
```

上传文档 → 输入问题 → 获取带来源标注的回答。

---

## 检索管线

```
用户提问
  │
  ├─→ HyDE         LLM 生成假设文档取代原始 query，缩小语义鸿沟
  │
  ├─→ 粗筛          FAISS 向量 (×24)  ─┐
  │                 BM25  稀疏 (×24)  ─┤  合并去重 → 候选池 ~50 条
  │                                    ┘
  ├─→ 精排          Cross-Encoder → 重打分 → Top 12
  │
  ├─→ 还原          子块命中 → 向上关联父块 → 动态取 4–8 块
  │
  └─→ 生成          Qwen-Max + 上下文 → 回答 + 溯源
```

---

## 功能详解

### 混合检索

单用向量检索容易遗漏精确术语匹配，单用关键词检索捕捉不到语义同义表达。两路并行召回后合并去重：

| 路径 | 实现 | 优势 |
|------|------|------|
| 向量 (Dense) | DashScope `text-embedding-async-v2` / BGE-M3 + FAISS | 同义表达、跨段落语义匹配 |
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

合并去重后的候选池（约 50 条）送入 Cross-Encoder 逐对计算 `(query, chunk)` 相关性分数，重排后截断至 Top 12。Cross-Encoder 比双塔 embedding 精细得多——它同时接收 query 和 chunk 作为输入做全注意力交叉编码。

| 后端 | 模型 | 适用场景 |
|------|------|----------|
| DashScope API | `gte-rerank-v2` | 免部署，100 万 token 免费额度 |
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
| `mvp_app.py` | 7860 | 日常问答：上传 + 对话一体化，启动快 |
| `gradio_app.py` | 7862 | 研究管理：批量导入 papers/、文档内容预览、文件清单表格 |

### 脚本工具

```bash
python scripts/fetch_papers.py --topic rag --max-per-topic 20   # arXiv 批量下载
python scripts/batch_import.py                                   # papers/ → 知识库
python scripts/download_bge.py                                   # BGE 本地模型（~3.2 GB）
```

### 检索调参

编辑 `config/faiss.yml`：

```yaml
vector_fetch_k: 24       # 向量路召回数
bm25_fetch_k: 24         # BM25 路召回数
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
├── config/
│   ├── settings.py              # 环境变量解析，全部配置入口
│   ├── faiss.yml                # 检索超参
│   └── rag.yml                  # 模型与生成参数
├── model/
│   ├── factory.py               # 模型工厂：API/本地切换 + LRU 缓存
│   └── dashscope_embedding.py   # DashScope Embedding 适配 LangChain 接口
├── rag/
│   ├── pipeline.py              # RAGPipeline 统一入口
│   ├── hybrid_retriever.py      # FAISS + BM25 双路召回 + Reranker 精排 + 动态 TopK
│   ├── hyde.py                  # HyDE 假设文档生成
│   ├── vector_store.py          # FAISS 索引 + 父子块持久化 + 文件清单
│   ├── document_loader.py       # PDF/TXT 加载 + 父子分块策略
│   └── generator.py             # Qwen OpenAI-compatible API 封装
├── scripts/
│   ├── fetch_papers.py          # arXiv 论文批量下载（多主题、自动去重、断点续传）
│   ├── batch_import.py          # 批量导入文档到知识库
│   ├── download_bge.py          # BGE-M3 + BGE-Reranker 模型下载（国内镜像优先）
│   └── download_models.py       # 通用 HuggingFace 模型下载器
├── prompts/
│   └── rag_summarize.txt        # RAG 系统 Prompt（含引用格式约束）
├── utils/                       # 日志、YAML 解析、路径工具
├── data/papers/                 # 知识库论文（已内置精选论文）
├── mvp_app.py                   # Gradio 精简版入口
├── gradio_app.py                # Gradio 完整版入口
└── requirements.txt
```

---

## 常见问题

<details>
<summary><b>如何获取 API Key？</b></summary>

访问 [阿里云百炼控制台](https://dashscope.aliyuncs.com/)，注册后进入「API-KEY 管理」创建。新用户赠送：text-embedding-async-v2（2000 万 token）、gte-rerank-v2（100 万 token）、qwen-max（100 万 token）。
</details>

<details>
<summary><b>支持哪些文件格式？</b></summary>

PDF 和 TXT。PDF 通过 PyPDFLoader 提取文本层；扫描版 PDF（纯图片）需先用 OCR 工具预处理为 TXT。
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
