<h1 align="center">RAG-Chinese-QA</h1>
<p align="center">
  Chinese document QA · evaluation engineering<br>
  Hybrid retrieval / reranking / inline citations / calibrated judge / traceable runs
</p>
<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-blue?logo=python" alt="Python">
  <img src="https://github.com/lhh737/rag-chinese-qa/actions/workflows/ci.yml/badge.svg" alt="CI">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/uv-managed-de5fe9" alt="uv">
</p>
<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>

## Overview

A RAG QA system over 106 Chinese papers/documents (32,501 chunks): FAISS + BM25 retrieval, RRF fusion, and reranking. Answers carry inline `[n]` citations; out-of-scope questions are refused.

The main work is the evaluation layer: 80 quality + 20 safety questions, a judge calibrated on 164 human blind labels, paired ablations, failure attribution, a second-domain transfer run, and a trace/registry record for every query.

The parts worth reading: eval-set and judge calibration, paired ablations and rejected changes, second-domain transfer, and annotation / QC / cost.

| | |
|---|---|
| Corpus | 106 docs / 32,501 chunks; 30 docs in the second domain |
| Eval sets | 80 quality + 20 safety; 30 in the second domain |
| Run registry | 28 runs (`eval_results/registry.jsonl`) |
| Tests | 90 offline tests + GitHub Actions |
| Models | qwen3.7-text-embedding / rerank; deepseek-flash for generation and judging |
| Cost | P0–P6 ≈ ¥31; second-domain run ¥0.20 |

> Personal project, not a production service; the focus is a reproducible evaluation record.

## Demo

<p align="center">
  <img src="assets/webui-qa.png" width="48%" alt="QA UI">
  <img src="assets/webui-refusal.png" width="48%" alt="Out-of-scope refusal">
</p>
<p align="center"><sub>Left: inline [n] citations and retrieval trace. Right: an out-of-scope question is refused.</sub></p>

## Results

<p align="center"><img src="assets/results.svg" width="760" alt="R1 baseline to holdout acceptance"></p>

| Metric | Paper dev 60 (R1) | Paper dev 60 (final) | Paper holdout 20 | E-commerce FAQ 30 |
|---|---:|---:|---:|---:|
| Recall@5 | 0.717 | 0.758 | 0.800 | 0.825* |
| False refusal | 23.1% | 23.1% | 16.7% | 23.1% |
| Answer correctness | 0.692 | 0.725 | 0.667 | 0.692 |
| Faithfulness / citation | 0.981 / 0.923 | 1.000 / 1.000 | 1.000 / 1.000 | 0.923 / 0.923 |
| Out-of-scope refusal | 100% | 100% | 100% | 100% |

R1 = HyDE on; final = R1 + HyDE off. *The second domain has only 30 documents, so its R@5 is not directly comparable to the 106-document paper corpus. It is shown to keep the same metric set.

### What did not survive testing

| Experiment | Result | Decision |
|---|---|---|
| Remove reranking | False refusal +13.5pp (p=0.023), correctness −12.3pp | Keep reranking |
| Enable HyDE | No quality gain, cost +44% | Turn off (DR-003) |
| RRF k scan k=10/30/60/100 | No significant difference | No conclusion |
| TopK 6–10 | False refusal p=0.131, multi-hop unchanged | Reject |
| Prompt v2 | Significantly worse (p=0.045) | Reject |
| Agentic retrieval (single-turn tool selection) | R@5 +1.7pp (within noise), MRR −13.1pp, multi-hop 8→10 | Reject (P9) |

## Evaluation design

<p align="center"><img src="assets/eval-loop.svg" width="760" alt="Evaluation workflow"></p>

| Stage | Output |
|---|---|
| Build | 80 + 20 questions; 5-type quota; dev 60 / holdout 20; hash + CHANGELOG |
| Run | Judged runner; trace / registry; per-question cost and latency |
| Inspect | `apps/review_viewer.html`: trace / blind labeling / spot-check / candidate review |
| Attribute | Rule engine + human spot-check (20/20) |
| Iterate | Paired ablations, McNemar, decision records |
| Safety | Three injection layers + harmful + boundary; pre/post defense comparison |

<p align="center"><img src="assets/reviewer-trace.png" width="760" alt="Review viewer: trace mode"></p>
<p align="center"><sub>Trace view: retrieval and fusion details, rerank scores, final context, generation and citations, cost breakdown.</sub></p>

## Second domain

| | Paper domain | E-commerce FAQ |
|---|---|---|
| Corpus | 106 docs / 32,501 chunks | 30 docs (orders + policies) |
| Questions | dev 60 + holdout 20 | 30 (all dev, mini eval) |
| Main failure mode | Retrieval 8/19 | Date arithmetic, order reference, policy branching, safe-but-unhelpful 8/26 |

With the same final config and judge, the second domain scores R@5 0.825, false refusal 23.1%, answer correctness 0.692, faithfulness/citation 0.923, and 100% out-of-scope refusal. The failure surface moves from “not retrieved” to “generation-side reasoning”, which is direct evidence that the question quota has to be redesigned per domain.

Details: [`docs/transferability.md`](docs/transferability.md).

## Judge calibration and consistency

| Check | Result |
|---|---|
| 164 human blind labels | 164/164 valid; TPR/TNR 1.000 |
| Second judge (20%) | 34/34 |
| Cross-family anchor | 10/10 |
| Second-domain transfer (20 questions / 56 judgments) | agreement 0.946, κ 0.825, TPR 1.000, TNR 0.936 |
| AI second annotation (same-family, 50 questions / 134 judgments) | agreement 0.948, κ 0.836; 2 AI-over-strict, 4 guideline gaps, 1 human miss |

The AI second annotator shares the model family with the judge/generator. Its report is an AI–human consistency check, not human inter-annotator κ; it can pre-screen, not replace human review.

## Architecture

<p align="center"><img src="assets/architecture.svg" width="760" alt="System architecture"></p>

- Retrieval: FAISS + BM25 → RRF → reranking; parent-child chunks (400-char child retrieval, 1,200-char parent context).
- Context: dynamic TopK 4–8; hard validation of `[n]` citations; refusal via prompt constraints + zero-hit fallback.
- One facade: CLI / Web UI / FastAPI / MCP / evaluation share `RAGPipeline`; no LangChain (DR-001).
- Observability: one trace per query with per-stage latency and cost; every run records its config fingerprint.

### Stack

| Layer | Tech |
|---|---|
| Retrieval | FAISS, BM25, RRF, qwen3.7-text-rerank |
| Generation | DeepSeek deepseek-flash |
| Serving | FastAPI (SSE), single-file Web UI, MCP 2026-07-28 |
| Evaluation | Custom judge, McNemar / Wilson, JSONL traces, Run Registry |
| Operations | SQLite QC dashboard, annotation SOP, badcase regression |

## Operations & cost

- Annotation/QC: [`docs/ops-playbook.md`](docs/ops-playbook.md) — 7 binary criteria, stratified 20% sampling, rework escalation, badcase promotion.
- Cost–quality: [`docs/cost-quality.md`](docs/cost-quality.md) — same-set dev 60: HyDE off keeps quality, −44% cost, −45% p50; removing reranking saves 84% but raises false refusals by 13.5pp.
- QC dashboard: [`eval_results/analysis/qc_dashboard.md`](eval_results/analysis/qc_dashboard.md) — 6 SQL queries, SQLite, no new dependencies.

## Quickstart

> Prerequisites: Python 3.12+, [uv](https://docs.astral.sh/uv/); DashScope (embedding / rerank) and DeepSeek (generation / judge) API keys.

```bash
git clone https://github.com/lhh737/rag-chinese-qa.git && cd rag-chinese-qa
uv sync && cp .env.example .env    # DASHSCOPE_API_KEY / GEN_API_KEY / JUDGE_*

uv run ragqa ingest                # build the index (106 docs, idempotent)
uv run ragqa serve                 # Web UI + FastAPI → http://127.0.0.1:8000
uv run pytest -q                   # 90 offline tests
```

Evaluation and calibration:

```bash
uv run python scripts/run_eval.py --suite quality --split dev --mode judged
uv run python scripts/calibrate_judge.py --segment tuning --labels <blind export>
uv run python scripts/attribute_run.py --run-id <run_id>
```

MCP installation, tools, and client registration: [`docs/mcp.md`](docs/mcp.md).

## Project structure

```
src/ragqa/            core package: retrieval / generation / evaluation / serving / MCP / tools
apps/                 Web UI + review viewer (single-file)
config/               retrieval / generation / pricing config
prompts/              answer / judge / annotator prompts
scripts/              evaluation, calibration, attribution, QC, badcase, second-domain scripts
eval_results/         run artifacts + analysis/ reports + regression/ tags
data/golden/          frozen datasets (golden_v1.1 / safety_v1)
data/business/        second-domain 30 questions + health + judge spot-check
data/business_faq/    second-domain corpus (local; SOURCES.md public, full text not committed)
docs/                 plans, guidelines, operations, cost, transferability, MCP
decisions/            DR-001 to DR-005
```

## Docs

| Document | Question it answers |
|---|---|
| [docs/transferability.md](docs/transferability.md) | Does the same evaluation method transfer to a business domain? |
| [docs/ops-playbook.md](docs/ops-playbook.md) | How are annotation, QC, rework, and badcase promotion run? |
| [docs/cost-quality.md](docs/cost-quality.md) | What does each quality point cost in money and latency? |
| [docs/annotation-guideline.md](docs/annotation-guideline.md) | How are the 7 criteria judged? |
| [docs/mcp.md](docs/mcp.md) | How to run and register the MCP server? |
| [docs/index.md](docs/index.md) | Full documentation index |
| [eval_results/analysis/](eval_results/analysis/) | Baseline, P6/P9, ablations, k-scan, QC dashboard |
| [decisions/](decisions/) | Design decisions and rejected changes |

## Limitations

- Single annotator; next-day relabel 26/30; the 164/164 zero-disagreement result is in-sample and the sample is small.
- The judge shares the model family with the generator; the cross-family anchor has only 10 samples; the second-domain judge check has 20 questions / 56 judgments.
- Retrieval metrics have ±3–5pp run-to-run noise; smaller differences are not interpreted.
- The second domain is a 30-question mini eval (all dev, mock data). b020/b021 lack an as-of date in the question; excluding them gives ac 18/24 = 0.750 and false refusal 4/24 = 16.7%.
- The multi-hop “second source never retrieved” gap is open. k / TopK / HyDE / prompt / multi-query did not close it; the remaining directions are chunking and embedding.

## Roadmap

- Close the multi-hop gap with chunking / embedding changes and paired re-validation.
- Expand safety-judge calibration; compare a constrained HyDE variant.
- Productionize `ragqa serve` (auth / rate limiting).
- Second-domain v1.1: add the missing as-of dates and re-run; re-run AI second annotation with the v1.1 prompt.

<p align="center"><sub>MIT License</sub></p>
