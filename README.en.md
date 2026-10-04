<h1 align="center">RAG-Chinese-QA</h1>
<p align="center">
  A Chinese-document RAG QA system built on LangChain<br>
  <b>Hybrid retrieval · HyDE · Reranking · Answer citation</b>
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

## Overview

**RAG-Chinese-QA** is an end-to-end RAG system for Chinese documents. Upload PDFs / TXT files and the system chunks, embeds and indexes them automatically — then ask questions in natural language, with every answer carrying citations back to the source text.

Compared to plain vector-search pipelines, the core difference is a **three-stage retrieval pipeline**: FAISS + BM25 dual-path coarse recall → cross-encoder reranking → parent-child context restoration, balancing recall and answer quality.

**API mode works out of the box with zero model downloads**; it can also switch to offline local BGE models.

---

## Quick Start

> Prerequisites: Python 3.10+, a [DashScope API key](https://dashscope.aliyuncs.com/) (Alibaba Cloud Model Studio — free quota on sign-up).

```bash
git clone https://github.com/lhh737/rag-chinese-qa.git
cd rag-chinese-qa

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # edit .env, set DASHSCOPE_API_KEY
python mvp_app.py                  # opens http://127.0.0.1:7860
```

Upload documents → ask questions → get answers with source citations.

---

## Retrieval Pipeline

```
User question
  │
  ├─→ HyDE         LLM expands the query into a hypothetical document,
  │                closing the semantic gap
  │
  ├─→ Coarse recall  FAISS dense (×24)   ─┐
  │                  BM25 sparse (×24)   ─┤  merge & dedup → ~50 candidates
  │                                      ┘
  ├─→ Rerank       Cross-Encoder → rescore → top 12
  │
  ├─→ Restore      child chunk hit → associate parent chunk → 4–8 dynamic blocks
  │
  └─→ Generate     Qwen-Max + context → answer + citations
```

---

## Features in Detail

### Hybrid retrieval

Dense retrieval alone misses exact terminology; keyword retrieval alone misses paraphrases. Both run in parallel, then merge and deduplicate:

| Path | Implementation | Strength |
|------|----------------|----------|
| Dense | DashScope `text-embedding-async-v2` / BGE-M3 + FAISS | Paraphrases, cross-paragraph semantics |
| Sparse | jieba tokenisation + BM25Okapi inverted index | Exact hits on names, model numbers, jargon |

### HyDE

User questions tend to be short and colloquial ("why does this model work well?"), while knowledge-base text is formal written prose. HyDE (*Hypothetical Document Embeddings*) first asks the LLM to expand the question into a textbook-style passage, then retrieves with that hypothetical document — aligning the query with the corpus's semantic space and measurably improving recall. Falls back to the original question if the LLM call fails.

### Parent-child chunking

Small chunks retrieve precisely but lose context; large chunks keep semantics but add noise. Parent-child chunking decouples the two:

| Level | Size | Role |
|-------|------|------|
| Child | ~400 chars | FAISS index + BM25 corpus — precise hits |
| Parent | ~1,200 chars | Associated on a child hit, restored for the LLM |

### Reranking

The merged candidate pool (~50) is scored pairwise by a cross-encoder computing `(query, chunk)` relevance, then truncated to top 12. Cross-encoders are far more precise than dual-tower embeddings — query and chunk are fed together through full cross-attention.

| Backend | Model | When to use |
|---------|-------|-------------|
| DashScope API | `gte-rerank-v2` | No deployment, 1M free tokens |
| Local | `BAAI/bge-reranker-base` | Offline, ~1GB GPU memory |

### Dynamic Top-K

The number of context blocks sent to the LLM adapts automatically: long questions get +2 blocks; when reranker scores cluster high, the cutoff loosens to cover detail branches. Final output is 4–8 blocks — a live balance between answer completeness and token cost.

### Answer citation

Every answer ends with a citation list naming the source file and quoting the original text (first 320 chars). The Gradio UI lets you click a document to view the full text and verify accuracy.

---

## Usage Guide

### Run modes

Two lines in `.env` switch between API and local models:

```ini
# API (default, zero download)   # Local (offline; run download_bge.py first)
EMBED_MODE=api                   EMBED_MODE=local
RERANK_MODE=api                  RERANK_MODE=local
```

### UI entry points

| File | Port | Purpose |
|------|------|---------|
| `mvp_app.py` | 7860 | Daily QA: upload + chat in one page, quick startup |
| `gradio_app.py` | 7862 | Research management: batch-import papers/, document preview, file listing |

### Scripts

```bash
python scripts/fetch_papers.py --topic rag --max-per-topic 20   # batch-download from arXiv
python scripts/batch_import.py                                   # papers/ → knowledge base
python scripts/download_bge.py                                   # local BGE models (~3.2 GB)
```

### Retrieval tuning

Edit `config/faiss.yml`:

```yaml
vector_fetch_k: 24       # dense recall count
bm25_fetch_k: 24         # BM25 recall count
rerank_top_n: 12         # kept after reranking
base_context_k: 4        # minimum context blocks
max_context_k: 8         # maximum context blocks
parent_chunk_size: 1200  # parent chunk size (chars)
child_chunk_size: 400    # child chunk size (chars)
```

- For **higher recall**: increase `vector_fetch_k` / `bm25_fetch_k`
- For **faster responses**: decrease `rerank_top_n` and `max_context_k`
- For **long-paragraph documents**: increase `parent_chunk_size` to avoid truncation

---

## Project Structure

```
├── config/
│   ├── settings.py              # Environment parsing — single config entry point
│   ├── faiss.yml                # Retrieval hyperparameters
│   └── rag.yml                  # Model and generation parameters
├── model/
│   ├── factory.py               # Model factory: API/local switch + LRU cache
│   └── dashscope_embedding.py   # DashScope embeddings adapted to LangChain
├── rag/
│   ├── pipeline.py              # RAGPipeline unified entry point
│   ├── hybrid_retriever.py      # FAISS + BM25 recall + reranker + dynamic top-k
│   ├── hyde.py                  # HyDE hypothetical document generation
│   ├── vector_store.py          # FAISS index + parent-child persistence + file listing
│   ├── document_loader.py       # PDF/TXT loading + parent-child chunking
│   └── generator.py             # Qwen OpenAI-compatible API wrapper
├── scripts/
│   ├── fetch_papers.py          # arXiv batch download (multi-topic, dedup, resume)
│   ├── batch_import.py          # Batch-import documents into the knowledge base
│   ├── download_bge.py          # BGE-M3 + BGE-Reranker download (CN mirrors first)
│   └── download_models.py       # Generic HuggingFace model downloader
├── prompts/
│   └── rag_summarize.txt        # RAG system prompt (with citation constraints)
├── utils/                       # Logging, YAML parsing, path helpers
├── data/papers/                 # Knowledge-base papers (curated set included)
├── mvp_app.py                   # Gradio minimal entry point
├── gradio_app.py                # Gradio full entry point
└── requirements.txt
```

---

## FAQ

<details>
<summary><b>How do I get an API key?</b></summary>

Go to the [Alibaba Cloud Model Studio console](https://dashscope.aliyuncs.com/) and create one under "API-KEY management". New users receive free quotas: text-embedding-async-v2 (20M tokens), gte-rerank-v2 (1M tokens), qwen-max (1M tokens).
</details>

<details>
<summary><b>Which file formats are supported?</b></summary>

PDF and TXT. PDFs are text-extracted with PyPDFLoader; scanned (image-only) PDFs must be OCR'd to TXT first.
</details>

<details>
<summary><b>Index errors after switching embedding models?</b></summary>

The system detects whether the FAISS dimension matches the current embedding model and rebuilds the index if not. If problems persist, delete the `faiss_db/` directory and restart.
</details>

<details>
<summary><b>Uploading large files is slow.</b></summary>

The bottleneck is the embedding API's QPS limit (free quota). Increase `child_chunk_size` to reduce the number of chunks and speed up indexing, at the cost of coarser retrieval granularity.
</details>

<details>
<summary><b>BM25 tokenisation seems off for Chinese.</b></summary>

The system uses jieba. Add custom words via `jieba.add_word("your-term")` in the `_zh_tokenize` function in `hybrid_retriever.py`.
</details>

---

## License

MIT — free to use, modify and distribute.
