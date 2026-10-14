<h1 align="center">RAG-Chinese-QA</h1>
<p align="center">
  Chinese-document RAG Q&A · <b>Evaluation-Centric Quality Engineering</b><br>
  Hybrid Retrieval (RRF) · Reranking · Parent-Child Chunks · Inline Citations [n] · <b>Custom Judge (TPR/TNR-calibrated)</b> · Attribution Loop
</p>
<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/uv-managed-de5fe9" alt="uv">
  <img src="https://img.shields.io/badge/tests-71%20passed-brightgreen" alt="tests">
  <img src="https://img.shields.io/badge/cost-%E2%89%A4%C2%A531%20total-informational" alt="cost">
</p>
<p align="center">
  <a href="README.en.md">English</a> | <a href="README.md">中文</a>
</p>

> This is not "yet another RAG demo" — it is **RAG quality engineering built around an evaluation
> system**. A six-stage loop (Build → Evaluate → Observe → Attribute → Iterate → Secure) is fully
> implemented, and every number traces back to a Run Registry and a decision record. Core belief:
> **measure first, optimize second** — the Judge was calibrated against 164 human blind labels
> (zero disagreement) before it was trusted for headline metrics; third-party tools (RAGAS) were
> validated before being admitted or rejected.

---

## Results at a glance (all reproducible under `eval_results/`)

<p align="center"><img src="assets/results.svg" width="760" alt="Results: baseline vs holdout; ablation findings"></p>

| Metric (quality set, dev 60 / holdout 20) | Baseline (R1) | **Holdout acceptance** |
|---|---|---|
| Recall@5 (document-level) | 0.717 | **0.800** |
| False-refusal rate | 0.231 | **0.167** |
| Answer correctness (judge) | 0.692 | **0.667** |
| Faithfulness / Citation accuracy (judge) | 0.981 / 0.923 | **1.000 / 1.000** |
| Out-of-scope refusal rate | 1.000 | 1.000 |

- **Judge calibration**: 164 human blind labels → **zero disagreement** with the calibrated judge;
  second-judge cross-checks 34/34; cross-family anchor on holdout 10/10
  (full TPR/TNR/κ in the [`P4 calibration report`](eval_results/calibration/reports/P4_calibration_report.md)).
- **Removal-based ablations (paired McNemar, 10 comparison runs)**: **reranking is the only
  significant contributor** (+13.5pp false refusals when removed, p=0.023); HyDE contributes ≈0
  (−45% cost → disabled by default, DR-003); RRF k-scan is a documented negative result; and we
  quantified a **±3–5pp run-to-run noise floor** for retrieval metrics.
- **Attribution**: a rule engine assigns 19 failures to 5 taxonomy classes; human spot-check
  agreement **20/20**; the biggest open gap (second-source recall on multi-hop) is documented
  honestly together with the negative results.
- **Tool validation**: RAGAS agreement with humans 50–65%, with three reproduced Chinese failure
  modes (refusal semantics / over-strict granularity / citation-blind) → rejected from the main
  pipeline; DeepEval intentionally reduced to pytest assertions. **Even the evaluation tools were
  validated first.**

---

## Web UI

<p align="center">
  <img src="assets/webui-qa.png" width="820" alt="Web UI: streamed answer with [n] citations + traceback panel (retrieval funnel / rerank scores / per-stage latency & cost)">
</p>

The UI shares the exact pipeline and trace with the evaluation system (same `RAGPipeline` facade):
streamed answers carry `[n]` citations that jump to the right-hand source blocks; the trace panel shows
the real retrieval funnel (vector/BM25 → RRF → rerank → dynamic TopK), rerank relevance scores and
per-stage latency / cost estimate. Out-of-corpus questions are **refused**, not answered from general
knowledge — the screenshot below shows both visibly low rerank scores (0.38/0.17/0.15/0.13) and the explicit refusal:

<p align="center">
  <img src="assets/webui-refusal.png" width="820" alt="Out-of-corpus question refused: rerank scores visibly low, answer declines explicitly">
</p>

## Architecture

<p align="center"><img src="assets/architecture.svg" width="760" alt="Architecture: retrieval pipeline and evaluation system"></p>

- **Parent-child chunks**: children (400 chars) for retrieval, parents (1200) for context;
- **Inline citations**: answer sentence ↔ `[n]` ↔ context block, parseable per sentence
  (with a code-level validity check on marker indices);
- **Single facade**: CLI / Web UI / FastAPI / evaluation engine share one `RAGPipeline`;
  fully LangChain-free (DR-001); the UI's traceback view is the same trace the evaluation reads.

## Evaluation system (the six-stage loop)

<p align="center"><img src="assets/eval-loop.svg" width="760" alt="The six-stage evaluation loop"></p>

| Stage | Deliverable |
|---|---|
| Build | `data/golden/`: 80 quality items (quota-stratified, dev/holdout) + 20 safety items (incl. split-across-chunks injection); annotation guideline `docs/annotation-guideline.md` |
| Evaluate | `src/ragqa/evaluation/`: Run Registry (config/version/cost traceability), judged runner, custom Judge (cache + drift protection) |
| Observe | `apps/review_viewer.html`: single-file viewer (review / blind-label / spot-check / attribution modes) |
| Attribute | `ragqa.evaluation.attribution`: taxonomy v1 rule engine (evidence-backed) + human spot-check loop |
| Iterate | Removal ablations / k-scan / prompt & config A-B tests decided by paired statistics; decision records `decisions/DR-00x` |
| Secure | Injection (query/doc/cross-chunk) + harmful + boundary suites; rule + judge layers; before/after defense comparison |

**Evaluation workbench** (`apps/review_viewer.html`, single-file, four modes: read-trace / blind-label /
spot-check / candidate-review) — error analysis (P3), judge-calibration labeling (P4) and attribution
spot-checks (P5) all happened on this workbench; every mode shares the same trace schema, and exports
feed straight into downstream analysis:

<p align="center">
  <img src="assets/reviewer-trace.png" width="820" alt="Review viewer · read-trace mode: recall (vector/BM25 top-8 with scores) and fusion details, rerank scores, final context, generation & citations, cost ledger + per-item open coding">
</p>

<p align="center">
  <img src="assets/reviewer-blind.png" width="820" alt="Review viewer · blind-label mode: judge verdicts hidden; binary per-criterion labeling against ground truth / key claims (with automatic citation-marker hints)">
</p>

## Quickstart

> Requires Python 3.12+, [uv](https://docs.astral.sh/uv/); API keys: DashScope (embeddings/rerank) + DeepSeek (generation/judge).

```bash
git clone https://github.com/lhh737/rag-chinese-qa.git && cd rag-chinese-qa
uv sync && cp .env.example .env    # fill DASHSCOPE_API_KEY / GEN_API_KEY / JUDGE_*

uv run ragqa ingest                # build index (105 papers + Chinese docs; idempotent, resumable)
uv run ragqa serve                 # Web UI + FastAPI → open http://127.0.0.1:8000
```

**Web UI** (`apps/web_ui.html`, single-file vanilla frontend, zero build — see "Web UI" above):
switchable prompt versions and embedded knowledge-base management (upload / batch import / view raw text).
API docs at `http://127.0.0.1:8000/docs`.

**Evaluation & calibration**:

```bash
uv run python scripts/run_eval.py --suite quality --split dev --mode judged   # full judged evaluation
uv run python scripts/calibrate_judge.py --segment tuning --labels <export>   # 3-segment judge calibration
uv run python scripts/attribute_run.py --run-id <run_id>                      # auto-attribution + spot-check file
uv run pytest -q                                                              # 71 offline tests
```

## Reproducibility & engineering discipline

- **Run Registry** (`eval_results/registry.jsonl`): type/config/version/metrics/cost per run;
  traces carry a full config fingerprint (`config_id`, e.g. `vec24_bm2524_rrf60_rr12_k4-8_hyde0_rr1bm1`).
- **Drift protection**: judge prompt or model change → re-run measurement-segment calibration;
  cache keys include a prompt content hash.
- **Cost**: full project (P0–P6) ≈ **¥31 total** (10 full judged runs + calibration + acceptance,
  including judge and rerank API costs).
- Frozen datasets / judge versions with recorded hashes (`data/golden/CHANGELOG.md`).

## Repository map

```
src/ragqa/            core package (core facade / retrieval / generation / evaluation / api)
scripts/              run & experiment toolkit: run_eval · calibrate_judge · attribute_run · judge_crosscheck · rejudge_errors
eval_results/         all run artifacts + analysis reports (analysis/: baseline & P6 reports, ablation panel, k-scan, RAGAS comparison)
decisions/            decision records (DR-001 dependency migration / DR-002 incident / DR-003 HyDE / DR-004 iteration rulings)
docs/                 eval plan · annotation guideline · label reference · P5 plan
data/golden/          frozen datasets (golden_v1.1 / safety_v1 + CHANGELOG + health report)
```

## Known limitations (stated honestly)

- **Single annotator**: intra-rater agreement 26/30 (next-day relabel); attribution spot-check 20/20 —
  small samples, wide intervals;
- **Judge**: primary judge shares the DeepSeek family with the generator (single-family fallback
  clause, self-preference limitation declared); cross-family anchor is sample-based only;
- **Noise floor**: ±3–5pp run-to-run variation on retrieval metrics; smaller deltas are not
  interpreted;
- **Open gap**: multi-hop "second source never retrieved" (candidate-pool-level). All four cheap
  levers (k, TopK, HyDE, prompt) were proven ineffective — this needs chunking / embedding /
  multi-query retrieval work (future direction);
- Safety judge is lightly calibrated (15 human cross-checks, see P6 report); an `s005`-type
  "abandons task due to injection" soft failure is recorded.

## Future work

Multi-hop retrieval-layer redesign (second query / source-aware chunking) with paired re-validation;
constrained-HyDE variant comparison; expanded safety-judge calibration; productionizing `ragqa serve`
(auth / rate limiting).

<p align="center"><sub>MIT License · Methodology inspired by Hamel Husain's "Your AI Product Needs Evals" et al.</sub></p>
