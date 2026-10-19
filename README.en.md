<h1 align="center">RAG-Chinese-QA</h1>
<p align="center">
  Chinese-document RAG Q&A · Evaluation-Centric Quality Engineering<br>
  Hybrid Retrieval (RRF) · Reranking · Parent-Chunk Context · Inline Citations [n] · Custom Judge (human-calibrated) · Attribution Loop
</p>
<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-blue?logo=python" alt="Python">
  <img src="https://github.com/lhh737/rag-chinese-qa/actions/workflows/ci.yml/badge.svg" alt="CI">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/uv-managed-de5fe9" alt="uv">
</p>
<p align="center">
  <a href="README.en.md">English</a> | <a href="README.md">中文</a>
</p>

A Chinese-document RAG Q&A system: hybrid retrieval (FAISS + BM25 → RRF) and reranking over 106 documents (32,501 chunks), answers with inline `[n]` citations, explicit refusal for out-of-corpus questions.

Around it, a full evaluation-centric loop — Build → Evaluate → Observe → Attribute → Iterate → Secure. The custom Judge was calibrated against 164 human blind labels before it was trusted for headline metrics; ablations, attributions and negative results are all on record, and every number traces back to the Run Registry and decision records.

---

## Results at a glance

<p align="center"><img src="assets/results.svg" width="760" alt="Key results: baseline vs holdout; ablation findings"></p>

| Metric (quality set, dev 60 / holdout 20) | Baseline (R1) | Holdout acceptance |
|---|---|---|
| Recall@5 (document-level) | 0.717 | 0.800 |
| False-refusal rate | 0.231 | 0.167 |
| Answer correctness (judge) | 0.692 | 0.667 |
| Faithfulness / Citation accuracy (judge) | 0.981 / 0.923 | 1.000 / 1.000 |
| Out-of-scope refusal rate | 1.000 | 1.000 |

- **Judge calibration**: 164 human blind labels → zero disagreement with the calibrated judge; second-judge cross-check 34/34; cross-family anchor on holdout 10/10 ([P4 calibration report](eval_results/calibration/reports/P4_calibration_report.md)).
- **Removal-based ablations** (paired McNemar): reranking is the only significant contributor (+13.5pp false refusals when removed, p=0.023); HyDE contributes ≈0 while costing 45% extra (disabled by default, DR-003); RRF k-scan is a documented negative result; ±3–5pp run-to-run noise floor on retrieval metrics.
- **Agentic retrieval control** (single-turn tool selection / multi-query merge, P9): the mechanism works as designed — the model issues 2.7 rewritten searches on average and the merged context spans sources — but it yields **no significant improvement on any quality criterion**: Recall@5 +1.7pp (inside the noise floor), **MRR −13.1pp (ranking degraded)**, false refusals +5.8pp, multi-hop false refusals 8→10; **the target gap stays open → not adopted** ([P9 report](eval_results/analysis/p9_report.md)).
- **Attribution**: 19 failures mapped to 5 taxonomy classes by a rule engine (evidence-backed), human spot-check agreement 20/20; the biggest gap — multi-hop second-source recall — is on record, unresolved.
- **Tool validation**: RAGAS agreement with humans 50–65%, with three reproduced Chinese failure modes → not admitted to the main pipeline; DeepEval reduced to pytest assertions.

## Web UI

The UI shares the exact pipeline and trace with the evaluation system (same `RAGPipeline`): streamed answers carry `[n]` citations that jump to the right-hand source blocks; the trace panel shows the real retrieval funnel, rerank scores and per-stage latency / cost.

<p align="center"><img src="assets/webui-qa.png" width="760" alt="Web UI"></p>
<p align="center"><sub>Answers with inline [n] citations; right panel: retrieval funnel, rerank scores, source texts</sub></p>

<p align="center"><img src="assets/webui-refusal.png" width="760" alt="Out-of-corpus refusal"></p>
<p align="center"><sub>Out-of-corpus question (coronary bypass) explicitly refused — rerank scores visibly low (0.13–0.38)</sub></p>

## Evaluation system

<p align="center"><img src="assets/eval-loop.svg" width="760" alt="The six-stage loop: Build → Evaluate → Observe → Attribute → Iterate → Secure"></p>

| Stage | Deliverable |
|---|---|
| Build | `data/golden/`: 80 quality items (quota-stratified, dev/holdout) + 20 safety items (incl. split-across-chunks injection); guideline `docs/annotation-guideline.md` |
| Evaluate | `src/ragqa/evaluation/`: Run Registry (config/version/cost traceability), judged runner, custom Judge (cache + drift protection) |
| Observe | `apps/review_viewer.html`: single-file viewer (review / blind-label / spot-check / candidate modes) |
| Attribute | `ragqa.evaluation.attribution`: taxonomy v1 rule engine + human spot-check loop |
| Iterate | Removal ablations / k-scan / prompt & config A-B tests decided by paired statistics; decision records `decisions/DR-00x` |
| Secure | Injection (query / doc / cross-chunk) + harmful + boundary suites; rule + judge layers; before/after defense comparison |

Evaluation workbench (`apps/review_viewer.html`): single file, four modes sharing one trace schema — error analysis (P3), judge-calibration labeling (P4) and attribution spot-checks (P5) all happened here:

<p align="center"><img src="assets/reviewer-trace.png" width="760" alt="Review viewer: read-trace mode"></p>
<p align="center"><sub>Read-trace: recall & fusion details, rerank scores, final context, generation & citations, cost ledger</sub></p>

<p align="center"><img src="assets/reviewer-blind.png" width="760" alt="Review viewer: blind-label mode"></p>
<p align="center"><sub>Blind-label: judge verdicts hidden; binary per-criterion labeling against ground truth</sub></p>

## Architecture

<p align="center"><img src="assets/architecture.svg" width="760" alt="Architecture: retrieval pipeline and evaluation system"></p>

**Context engineering** — what enters the generator's context window, how much of it, and under what constraints is decided explicitly by the retrieval layer, and every trade-off carries ablation evidence:

- Parent-child chunks: children (400 chars) for retrieval, parents (1200 chars) restore context; split granularity and restoration are configurable;
- Context budget: dynamic TopK (4–8) decides how many parents enter the window, driven by the rerank score distribution;
- Inline citation constraint: answer sentence ↔ `[n]` ↔ context block, parseable per sentence, with a code-level validity check on marker indices;
- Refusal policy: out-of-corpus questions are decided by prompt constraints plus a zero-hit fallback — never silently fabricated;
- Evidence: reranking is the only significant contributor (+13.5pp false refusals when removed, p=0.023); HyDE contributes ≈0 so it is off, saving 45% cost (DR-003); Prompt v2 was rejected (degraded, p=0.045).
- Single facade: CLI / Web UI / FastAPI / MCP server / evaluation engine share one `RAGPipeline`; fully LangChain-free (DR-001), and the MCP tools are thin wrappers with no separate retrieval logic.

## Quickstart

> Requires Python 3.12+, [uv](https://docs.astral.sh/uv/); API keys: DashScope (embeddings / rerank) + DeepSeek (generation / judge).

```bash
git clone https://github.com/lhh737/rag-chinese-qa.git && cd rag-chinese-qa
uv sync && cp .env.example .env    # fill DASHSCOPE_API_KEY / GEN_API_KEY / JUDGE_*

uv run ragqa ingest                # build index (106 documents; idempotent, resumable)
uv run ragqa serve                 # Web UI + FastAPI → http://127.0.0.1:8000
```

The Web UI (`apps/web_ui.html`, single-file vanilla frontend, zero build) supports prompt-version switching and knowledge-base management (upload / batch import / view raw text); API docs at `http://127.0.0.1:8000/docs`.

MCP server (`ragqa mcp`): exposes retrieval / raw-text / QA as standard MCP tools for any MCP client. Built on MCP spec **2026-07-28** with Python SDK v2 (`MCPServer`); requires the optional extra:

```bash
uv sync --extra mcp
uv run ragqa mcp                                # stdio (launched by the client; stdout carries protocol messages only)
uv run ragqa mcp --transport http --port 8000   # Streamable HTTP → http://127.0.0.1:8000/mcp
```

Tools: `search_documents` / `get_document_content` / `list_documents` / `answer` — thin wrappers over the same `RAGPipeline`, identical to CLI / Web UI / evaluation engine. Client registration (stdio):

```json
{"mcpServers": {"rag-chinese-qa": {"type": "stdio", "command": "uv",
  "args": ["run", "--directory", "<repo-path>", "ragqa", "mcp"]}}}
```

Or register directly: `claude mcp add --scope project ragqa -- uv run --directory <repo-path> ragqa mcp`

Evaluation & calibration:

```bash
uv run python scripts/run_eval.py --suite quality --split dev --mode judged   # full judged evaluation
uv run python scripts/calibrate_judge.py --segment tuning --labels <export>   # 3-segment judge calibration
uv run python scripts/attribute_run.py --run-id <run_id>                      # auto-attribution + spot-check file
uv run pytest -q                                                              # 90 offline tests
```

## Observability & reproducibility

- **End-to-end trace** (`src/ragqa/trace.py`, Trace Schema v1): one JSONL record per query covering HyDE / vector / BM25 / RRF / rerank / context / generation / usage, with per-stage latency and per-stage cost; evaluation, error analysis, attribution and the Web UI trace panel all read the same schema.
- **Run Registry** (`eval_results/registry.jsonl`): type / config / version / metrics / cost per run; traces carry a full config fingerprint (`config_id`, e.g. `vec24_bm2524_rrf60_rr12_k4-8_hyde0_rr1bm1`).
- **Serving side**: `/health` exposes index metadata (documents / dim / embedding model / pipeline version) and the config fingerprint.
- **Drift protection**: judge prompt or model change → re-run measurement-segment calibration (cache keys include a prompt content hash); datasets and judge versions frozen with recorded hashes (`data/golden/CHANGELOG.md`).
- **Cost**: full project (P0–P6) ≈ ¥31 total (10 full judged runs + calibration + acceptance).

## Operations & cost

- **Annotation/QC SOP**: `docs/ops-playbook.md` — 7 binary criteria, stratified 20% sampling, rework escalation, AI second-annotation protocol, badcase promotion.
- **Cost–quality–latency**: `docs/cost-quality.md` — same-set dev 60: HyDE off keeps quality, −44% cost, −45% p50; removing rerank saves 84% but raises false refusals by 13.5pp (p=0.023).
- **QC dashboard**: `eval_results/analysis/qc_dashboard.md` (SQL: `qc_dashboard.sql`; reproduce: `uv run python scripts/qc_metrics.py`, SQLite, no new dependencies).
- **Badcase promotion**: `scripts/promote_badcase.py` writes only to `eval_results/regression/`, never edits golden; new items require human review + CHANGELOG.

## Second domain & AI second annotation

- **Second-domain mini eval (e-commerce customer service FAQ)**: same final config / judge, 30 questions on the public `ecom-cs-agent` dataset (mock orders/logistics + policies). R@5 0.825, false refusal 23.1%, answer correctness 0.692; the failure surface moved from retrieval (8/19 in the paper domain) to date arithmetic, order reference, policy branching, and safe-but-unhelpful refusals (8/26). Report: [`docs/transferability.md`](docs/transferability.md).
- **AI second annotation (same-family, honestly labeled)**: 50 questions / 134 judgments; AI–human agreement 0.948, κ 0.836; 7 disagreements adjudicated as 2 AI-over-strict, 4 guideline gaps, 1 human miss — AI is a pre-screen, not a replacement for human review. Report: [`P4b_ai_second_annotation.md`](eval_results/calibration/reports/P4b_ai_second_annotation.md).
- **Judge transfer spot-check (complete)**: 20 questions / 56 human judgments; judge–human agreement 0.946, κ 0.825, TPR 1.000, TNR 0.936; 3 disagreements. Report: [`judge_spotcheck_v1_report.md`](data/business/judge_spotcheck_v1_report.md).

## Repository map

```
src/ragqa/            core package (core facade / retrieval / generation / evaluation / api / tools layer / mcp_server / agentic single-turn tool selection)
apps/                 Web UI (web_ui.html) + review viewer (review_viewer.html), both single-file
scripts/              evaluation & experiment scripts: run_eval · calibrate_judge · attribute_run · judge_crosscheck · rejudge_errors · qc_metrics · promote_badcase · build_business_* · ai_second_annotate
eval_results/         all run artifacts + analysis reports (analysis/: baseline & P6/P9 reports, ablation panel, k-scan, RAGAS comparison, QC dashboard; regression/: badcase regression tags)
decisions/            decision records (DR-001 dependency migration / DR-002 incident / DR-003 HyDE / DR-004 iteration rulings / DR-005 MCP dependency)
docs/                 eval plan · annotation guideline · label reference · P5 plan · ops playbook · cost-quality · transferability
data/golden/          frozen datasets (golden_v1.1 / safety_v1 + CHANGELOG + health report)
data/business/        second-domain mini eval (30 questions + health + judge spot-check)
data/business_faq/    second-domain corpus (local; SOURCES.md tracked, full text gitignored)
```

## Known limitations

- Single annotator: next-day relabel agreement 26/30; attribution spot-check 20/20 — small samples, wide intervals;
- Judge: primary judge shares the DeepSeek family with the generator (single-family fallback, self-preference limitation declared); cross-family anchor is sample-based only;
- Noise floor: ±3–5pp run-to-run variation on retrieval metrics; smaller deltas are not interpreted;
- Open gap: multi-hop "second source never retrieved" (candidate-pool level) — all four cheap levers (k / TopK / HyDE / prompt) were proven ineffective, and **multi-query retrieval (single-turn tool selection, P9) failed too**; the remaining directions narrow to chunking strategy / embedding model;
- Safety judge is lightly calibrated (15 human cross-checks); an `s005`-type "abandons task due to injection" soft failure is on record.

## Future work

Multi-hop gap redesign (chunking strategy / embedding model — multi-query retrieval was falsified in P9) with paired re-validation; constrained-HyDE variant comparison; expanded safety-judge calibration; productionizing `ragqa serve` (auth / rate limiting).

<p align="center"><sub>MIT License · Methodology inspired by Hamel Husain's "Your AI Product Needs Evals" et al.</sub></p>
