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

- Parent-child chunks: children (400 chars) for retrieval, parents (1200 chars) restore context;
- Inline citations: answer sentence ↔ `[n]` ↔ context block, parseable per sentence (with a code-level validity check on marker indices);
- Single facade: CLI / Web UI / FastAPI / evaluation engine share one `RAGPipeline`; fully LangChain-free (DR-001).

## Quickstart

> Requires Python 3.12+, [uv](https://docs.astral.sh/uv/); API keys: DashScope (embeddings / rerank) + DeepSeek (generation / judge).

```bash
git clone https://github.com/lhh737/rag-chinese-qa.git && cd rag-chinese-qa
uv sync && cp .env.example .env    # fill DASHSCOPE_API_KEY / GEN_API_KEY / JUDGE_*

uv run ragqa ingest                # build index (106 documents; idempotent, resumable)
uv run ragqa serve                 # Web UI + FastAPI → http://127.0.0.1:8000
```

The Web UI (`apps/web_ui.html`, single-file vanilla frontend, zero build) supports prompt-version switching and knowledge-base management (upload / batch import / view raw text); API docs at `http://127.0.0.1:8000/docs`.

Evaluation & calibration:

```bash
uv run python scripts/run_eval.py --suite quality --split dev --mode judged   # full judged evaluation
uv run python scripts/calibrate_judge.py --segment tuning --labels <export>   # 3-segment judge calibration
uv run python scripts/attribute_run.py --run-id <run_id>                      # auto-attribution + spot-check file
uv run pytest -q                                                              # 71 offline tests
```

## Engineering & reproducibility

- **Run Registry** (`eval_results/registry.jsonl`): type / config / version / metrics / cost per run; traces carry a full config fingerprint (`config_id`, e.g. `vec24_bm2524_rrf60_rr12_k4-8_hyde0_rr1bm1`).
- **Drift protection**: judge prompt or model change → re-run measurement-segment calibration (cache keys include a prompt content hash); datasets and judge versions frozen with recorded hashes (`data/golden/CHANGELOG.md`).
- **Cost**: full project (P0–P6) ≈ ¥31 total (10 full judged runs + calibration + acceptance).

## Repository map

```
src/ragqa/            core package (core facade / retrieval / generation / evaluation / api)
apps/                 Web UI (web_ui.html) + review viewer (review_viewer.html), both single-file
scripts/              evaluation & experiment scripts: run_eval · calibrate_judge · attribute_run · judge_crosscheck · rejudge_errors, etc.
eval_results/         all run artifacts + analysis reports (analysis/: baseline & P6 reports, ablation panel, k-scan, RAGAS comparison)
decisions/            decision records (DR-001 dependency migration / DR-002 incident / DR-003 HyDE / DR-004 iteration rulings)
docs/                 eval plan · annotation guideline · label reference · P5 plan
data/golden/          frozen datasets (golden_v1.1 / safety_v1 + CHANGELOG + health report)
```

## Known limitations

- Single annotator: next-day relabel agreement 26/30; attribution spot-check 20/20 — small samples, wide intervals;
- Judge: primary judge shares the DeepSeek family with the generator (single-family fallback, self-preference limitation declared); cross-family anchor is sample-based only;
- Noise floor: ±3–5pp run-to-run variation on retrieval metrics; smaller deltas are not interpreted;
- Open gap: multi-hop "second source never retrieved" (candidate-pool level) — all four cheap levers (k / TopK / HyDE / prompt) were proven ineffective; needs chunking / embedding / multi-query retrieval work;
- Safety judge is lightly calibrated (15 human cross-checks); an `s005`-type "abandons task due to injection" soft failure is on record.

## Future work

Multi-hop retrieval-layer redesign (second query / source-aware chunking) with paired re-validation; constrained-HyDE variant comparison; expanded safety-judge calibration; productionizing `ragqa serve` (auth / rate limiting).

<p align="center"><sub>MIT License · Methodology inspired by Hamel Husain's "Your AI Product Needs Evals" et al.</sub></p>
