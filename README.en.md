<h1 align="center">RAG-Chinese-QA</h1>
<p align="center">
  A Chinese-document RAG QA system · <b>evaluation-first quality engineering (upgrade in progress)</b><br>
  <b>Hybrid retrieval (RRF) · HyDE · Reranking · Parent-child chunks · Inline [n] citations</b>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/uv-managed-de5fe9" alt="uv">
</p>
<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>

> 🚧 **Upgrade in progress**: the project is being upgraded into an evaluation-first RAG quality system (construct → evaluate → observe → attribute → iterate → secure).
> P0 done: `src/ragqa` restructure (unified facade + structured traces + LangChain-free dependencies + full index rebuild); eval framework and datasets under construction. README will be fully rewritten at the final stage.

---

## Overview

**RAG-Chinese-QA** is an end-to-end RAG system for Chinese documents. Upload PDFs / TXT files and the system chunks, embeds and indexes them automatically — then ask questions in natural language, with every answer carrying citations back to the source text.

Compared to plain vector-search pipelines, the core difference is a **three-stage retrieval pipeline**: FAISS + BM25 dual-path coarse recall → cross-encoder reranking → parent-child context restoration, balancing recall and answer quality.

**API mode works out of the box with zero model downloads**; it can also switch to offline local BGE models.

---

## Quick Start

> Prerequisites: Python 3.11+ and [uv](https://docs.astral.sh/uv/); two API keys: [DashScope](https://dashscope.aliyuncs.com/) (embeddings / rerank) and [DeepSeek](https://platform.deepseek.com/) (generation).

```bash
git clone https://github.com/lhh737/rag-chinese-qa.git
cd rag-chinese-qa

uv sync                            # create .venv and install dependencies
cp .env.example .env               # set DASHSCOPE_API_KEY and GEN_API_KEY

uv run ragqa ingest                # build the index (105 papers + Chinese docs; idempotent, resumable)
uv run python apps/gradio_ui.py    # opens http://127.0.0.1:7860
```

Upload documents → ask questions → get answers with inline `[n]` citations and sources.

---

## Retrieval Pipeline

```
User question
  │
  ├─→ HyDE         LLM expands the query into a hypothetical document,
  │                closing the semantic gap
  │
  ├─→ Coarse recall  FAISS dense (×24)   ─┐
  │                  BM25 sparse (×24)   ─┤  RRF fusion (k configurable) → ~48 candidates
  │                                      ┘
  ├─→ Rerank       Cross-Encoder → rescore → top 12
  │
  ├─→ Restore      child chunk hit → associate parent chunk → 4–8 dynamic blocks
  │
  └─→ Generate     DeepSeek + context → answer + inline [n] citations
```

---

## Features in Detail

### Hybrid retrieval

Dense retrieval alone misses exact terminology; keyword retrieval alone misses paraphrases. Both run in parallel, fused with **RRF (Reciprocal Rank Fusion)**:

| Path | Implementation | Strength |
|------|----------------|----------|
| Dense | DashScope `qwen3.7-text-embedding-flash` / BGE-M3 + faiss-cpu | Paraphrases, cross-paragraph semantics |
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

The RRF-fused candidate pool (~48) is scored pairwise by the rerank model computing `(query, chunk)` relevance, then truncated to top 12. Rerankers are far more precise than dual-tower embeddings — query and chunk are fed together through cross-attention.

| Backend | Model | When to use |
|---------|-------|-------------|
| DashScope API | `qwen3.7-text-rerank` | No deployment, pay-as-you-go |
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
| `apps/gradio_ui.py` | 7860 | Single entry: upload / batch-import papers/, chat with [n] citations, document viewer |

### CLI

```bash
uv run ragqa ingest                     # papers/ → knowledge base (idempotent, resumable)
uv run ragqa documents                  # list indexed documents
uv run ragqa query "What is RAG?"       # single query (--trace-out to persist the trace)
uv run ragqa smoke                      # smoke: small index + 5 queries + Recall@5
python scripts/fetch_papers.py --topic rag --max-per-topic 20   # batch-download from arXiv
python scripts/download_bge.py          # local BGE models (~3.2 GB, optional offline mode)
```

### Retrieval tuning

Edit `config/retrieval.yml`:

```yaml
vector_fetch_k: 24       # dense recall count
bm25_fetch_k: 24         # BM25 recall count
fusion: rrf              # fusion strategy (rrf | merge)
rrf_k: 60                # RRF parameter
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
├── src/ragqa/
│   ├── core.py                  # RAGPipeline unified facade (evals / UI / API share it)
│   ├── cli.py                   # CLI: ingest / documents / query / smoke
│   ├── trace.py                 # Trace Schema v1 (JSONL) + per-stage cost accounting
│   ├── types.py                 # Core dataclasses (Document / ContextBlock / Citation…)
│   ├── config/                  # pydantic-settings + YAML config loading
│   ├── ingestion/               # Parsing (pypdf) + sanitising + custom splitter + pipeline
│   ├── retrieval/               # direct faiss + dual-path recall (RRF) + rerank + HyDE
│   ├── generation/              # prompt registry + [n] citation parsing + generation/streaming
│   └── models/                  # Model factory: DashScope embeddings/rerank + DeepSeek generation
├── apps/gradio_ui.py            # Gradio single entry (7860)
├── prompts/answer/v1.md         # Answer prompt (versioned registry)
├── config/
│   ├── retrieval.yml            # Retrieval hyperparameters (RRF k / fusion / top-k / chunking)
│   ├── generation.yml           # Generation parameters
│   └── pricing.yml              # Cost estimation rates (CNY per 1M tokens)
├── tests/                       # Offline pytest suite (fake embeddings, no API key needed)
├── scripts/                     # Paper download + dependency-migration parity experiments
├── data/papers/                 # Knowledge-base papers (curated set included)
├── eval_results/                # Run registry and reports (under construction, P1+)
├── decisions/                   # Decision records (DR)
└── pyproject.toml  uv.lock
```

---

## FAQ

<details>
<summary><b>How do I get an API key?</b></summary>

Two keys are needed: ① create one under "API-KEY management" in the [Alibaba Cloud Model Studio console](https://dashscope.aliyuncs.com/) (for embeddings / rerank); ② create one on the [DeepSeek platform](https://platform.deepseek.com/) (for generation). Both are pay-as-you-go with free quotas for new users.
</details>

<details>
<summary><b>Which file formats are supported?</b></summary>

PDF and TXT. PDFs are text-extracted with pypdf; scanned (image-only) PDFs must be OCR'd to TXT first.
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
