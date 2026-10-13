"""P4 Judge 校准（设计 §4.5）：在盲标集上运行裁判，与人工盲标逐条对账（TPR/TNR/κ）。

三段纪律（`calibration_segments.json`）：
- 示例段 example    ：few-shot 来源，不参与测量（可跑，用于核对 few-shot 与人工标签一致性）；
- 调参段 tuning     ：迭代 judge prompt（≤2 轮）；
- 测量段 measurement：最终 TPR/TNR——调参期间不得运行/查看（须显式 --allow-measurement）。

用法：
    uv run python scripts/calibrate_judge.py --segment tuning --labels <盲标导出.jsonl>
    uv run python scripts/calibrate_judge.py --segment example --labels <盲标导出.jsonl>
    uv run python scripts/calibrate_judge.py --segment measurement --allow-measurement --labels <盲标导出.jsonl>

输入：
- `eval_results/calibration/blind_label_v1.jsonl`（题 + 基线答案 + 上下文；判定材料与此完全一致）
- 盲标导出（查看器模式②；每行 {"id","criterion","label","reason"}；缺省自动取最新 *_blind_*.jsonl）

输出：
- `eval_results/calibration/runs/calib_<segment>_<ts>.jsonl`（逐条判定 + 人标对照）
- `eval_results/calibration/reports/calib_<segment>_<ts>.md`（TPR/TNR/混淆矩阵/一致率/κ + 分歧清单）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ragqa.config.settings import get_settings  # noqa: E402
from ragqa.evaluation.judges.cache import JudgeCache  # noqa: E402
from ragqa.evaluation.judges.client import JudgeClient, JudgeError  # noqa: E402
from ragqa.evaluation.metrics import proportion  # noqa: E402
from ragqa.models.factory import get_judge_client  # noqa: E402

MARKER_RE = re.compile(r"\[(\d{1,3})\]|【(\d{1,3})】")
QUALITY_CRITERIA = ["answer_correctness", "faithfulness", "citation_accuracy", "oos_refusal"]
SEG_NAME = {"示例段": "example", "调参段": "tuning", "测量段": "measurement"}
SEG_CN = {v: k for k, v in SEG_NAME.items()}


# ── 判定材料（与查看器所选完全同源）─────────────────────────

def build_payload(item: dict) -> dict:
    contexts = item.get("contexts") or []
    context_numbered = "\n\n".join(
        f"[{c.get('rank', i + 1)}]（{c.get('source', '')}）\n{c.get('text', '')}"
        for i, c in enumerate(contexts)
    )
    return {
        "question": item.get("question", ""),
        "answer": item.get("answer", ""),
        "context": "\n\n".join(c.get("text", "") for c in contexts),
        "context_numbered": context_numbered,
        "n_contexts": len(contexts),
        "ground_truth": item.get("ground_truth", ""),
        "key_claims": "\n".join(f"- {c}" for c in (item.get("key_claims") or [])),
        "expected_behavior": item.get("expected_behavior", ""),
        "injection_marker": item.get("injection_marker", "") or "",
    }


def code_citation_check(item: dict) -> dict:
    """citation_accuracy 代码层（设计 §4.3）：编号越界检测。"""
    n = len(item.get("contexts") or [])
    invalid = []
    for m in MARKER_RE.finditer(item.get("answer") or ""):
        num = int(m.group(1) or m.group(2))
        if not (1 <= num <= n):
            invalid.append(num)
    return {"invalid_markers": sorted(set(invalid)), "n_contexts": n, "pass": not invalid}


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """人工 vs 裁判 二分类 κ（样本退化时返回 None）。"""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for h, j in pairs if h == j) / n
    pe = sum(
        (sum(1 for h, _ in pairs if h == L) / n) * (sum(1 for _, j in pairs if j == L) / n)
        for L in ("pass", "fail")
    )
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def metrics(pairs: list[tuple[str, str]]) -> dict:
    tp = sum(1 for h, j in pairs if h == "fail" and j == "fail")
    fn = sum(1 for h, j in pairs if h == "fail" and j == "pass")
    tn = sum(1 for h, j in pairs if h == "pass" and j == "pass")
    fp = sum(1 for h, j in pairs if h == "pass" and j == "fail")
    return {
        "n": len(pairs),
        "tpr": proportion(tp, tp + fn) if (tp + fn) else None,
        "tnr": proportion(tn, tn + fp) if (tn + fp) else None,
        "agreement": proportion(tp + tn, len(pairs)) if pairs else None,
        "kappa": cohen_kappa(pairs),
        "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp,
                      "human_fail": tp + fn, "human_pass": tn + fp},
    }


def fmt_pct(p: dict | None) -> str:
    if not p:
        return "—（无样本）"
    return f"{p['value']:.3f} [{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}] ({p['k']}/{p['n']})"


def fmt_kappa(k: float | None) -> str:
    return "—" if k is None else f"{k:.3f}"


# ── 主流程 ─────────────────────────────────────────────────

def _load_labels(path: Path) -> dict[tuple[str, str], dict]:
    labels: dict[tuple[str, str], dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            labels[(r["id"], r["criterion"])] = {"label": r.get("label"), "reason": r.get("reason", "")}
    return labels


def _auto_labels(calib_dir: Path) -> Path | None:
    cands = sorted(calib_dir.glob("*_blind_*.jsonl"))
    return cands[-1] if cands else None


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=str(REPO_ROOT / "eval_results/calibration/blind_label_v1.jsonl"))
    ap.add_argument("--segments-file", default=str(REPO_ROOT / "eval_results/calibration/calibration_segments.json"))
    ap.add_argument("--labels", default=None, help="盲标导出 JSONL；缺省自动取 calibration/ 下最新 *_blind_*.jsonl")
    ap.add_argument("--segment", default="tuning", choices=["example", "tuning", "measurement", "all"])
    ap.add_argument("--criteria", default=",".join(QUALITY_CRITERIA))
    ap.add_argument("--judge-model", default=None, help="缺省取 .env JUDGE_MODEL")
    ap.add_argument("--prompt-version", default="v1")
    ap.add_argument("--max-tokens", type=int, default=2048, help="思考型裁判需余量（思考 token 与 JSON 共享预算）")
    ap.add_argument("--think-off", action="store_true", help="尝试关闭思考（仅部分模型支持；不支持会报错）")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--allow-measurement", action="store_true", help="测量段纪律开关（调参期间不得使用）")
    args = ap.parse_args()

    if args.segment == "measurement" and not args.allow_measurement:
        print("⚠ 测量段纪律：调参期间不得运行/查看测量段。最终定稿时加 --allow-measurement 再跑。")
        return 2
    segments_wanted = list(SEG_NAME.values()) if args.segment == "all" else [args.segment]

    # ── 载入 ──
    items: dict[str, dict] = {}
    with Path(args.items).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                it = json.loads(line)
                items[it["id"]] = it
    seg_map = {
        qid: SEG_NAME.get(cn)
        for qid, cn in json.loads(Path(args.segments_file).read_text(encoding="utf-8"))["segments"].items()
    }
    labels_path = Path(args.labels) if args.labels else _auto_labels(Path(args.items).parent)
    if not labels_path or not labels_path.is_file():
        print("✗ 找不到盲标导出文件（--labels 或 calibration/*_blind_*.jsonl）；请先从查看器导出。")
        return 2
    labels = _load_labels(labels_path)
    print(f"题 {len(items)} · 标注 {len(labels)} 条 · 来源 {labels_path.name}")

    criteria = [c for c in args.criteria.split(",") if c]
    tasks = [
        (it, cr)
        for qid, it in items.items()
        if seg_map.get(qid) in segments_wanted
        for cr in (it.get("criteria") or [])
        if cr in criteria
    ]
    tasks.sort(key=lambda t: (SEG_NAME.get(seg_map.get(t[0]["id"]), ""), t[0]["id"], t[1]))
    print(f"待判定 {len(tasks)} 条（题 × criterion）")

    s = get_settings()
    model = args.judge_model or s.judge_model
    if not model:
        print("✗ 未配置裁判模型（.env JUDGE_MODEL 或 --judge-model）")
        return 2
    extra_body: dict = {"thinking": {"type": "disabled"}} if args.think_off else {}
    judge = JudgeClient(
        model=model, cache=JudgeCache(), client=get_judge_client(),
        extra_body=extra_body, max_tokens=args.max_tokens,
    )
    print(f"裁判模型：{model}（max_tokens={args.max_tokens}{'，think-off' if args.think_off else ''}）")

    def run_one(task: tuple[dict, str]) -> dict:
        it, cr = task
        row = {
            "id": it["id"], "criterion": cr, "segment": seg_map.get(it["id"]),
            "type": it.get("type"), "human": labels.get((it["id"], cr)),
        }
        if cr == "citation_accuracy":
            row["code"] = code_citation_check(it)
        try:
            row["judge"] = judge.judge(
                item_id=it["id"], answer=it.get("answer", ""), criterion=cr,
                prompt_version=args.prompt_version, payload=build_payload(it),
            )
        except JudgeError as e:
            row["judge"] = {"verdict": "error", "error": str(e)}
        return row

    t0 = time.time()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for i, row in enumerate(ex.map(run_one, tasks), start=1):
            results.append(row)
            if i % 20 == 0 or i == len(tasks):
                print(f"  … {i}/{len(tasks)}")

    # composed 合成：citation_accuracy = judge 语义层 ∪ 代码层（编号越界）
    for r in results:
        jv = r["judge"].get("verdict")
        if r["criterion"] == "citation_accuracy" and jv in ("pass", "fail"):
            r["composed"] = "fail" if (jv == "fail" or not r["code"]["pass"]) else "pass"
        else:
            r["composed"] = jv if jv in ("pass", "fail") else None

    def pairs_for(seg: str, crit: str, use_composed: bool) -> list[tuple[str, str]]:
        out = []
        for r in results:
            if seg != "all" and r["segment"] != seg:
                continue
            if r["criterion"] != crit:
                continue
            h = (r.get("human") or {}).get("label")
            v = r.get("composed") if use_composed else r["judge"].get("verdict")
            if h in ("pass", "fail") and v in ("pass", "fail"):
                out.append((h, v))
        return out

    segs_report = list(segments_wanted)
    if args.segment == "all" and len(segments_wanted) > 1:
        segs_report.append("all")
    report_m = {seg: {cr: metrics(pairs_for(seg, cr, True)) for cr in criteria} for seg in segs_report}
    judge_only_m = {seg: {cr: metrics(pairs_for(seg, cr, False)) for cr in criteria} for seg in segs_report}

    # ── 输出 ──
    ts = time.strftime("%Y%m%d-%H%M%S")
    runs_dir = Path(args.items).parent / "runs"
    reports_dir = Path(args.items).parent / "reports"
    runs_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    raw_path = runs_dir / f"calib_{args.segment}_{ts}.jsonl"
    with raw_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_err = sum(1 for r in results if r["judge"].get("verdict") == "error")
    n_cached = sum(1 for r in results if r["judge"].get("_cached"))
    n_skip = sum(
        1 for r in results if (r.get("human") or {}).get("label") not in ("pass", "fail")
    )
    tok_in = sum((r["judge"].get("usage") or {}).get("prompt_tokens", 0) for r in results)
    tok_out = sum((r["judge"].get("usage") or {}).get("completion_tokens", 0) for r in results)

    L: list[str] = []
    L.append(f"# Judge 校准报告 · {args.segment} · {ts}\n")
    L.append(f"- 裁判模型：`{model}` · prompt `{args.prompt_version}` · max_tokens {args.max_tokens}")
    L.append(f"- 标注来源：`{labels_path.name}` · 判定 {len(results)} 条（error {n_err} / 缓存命中 {n_cached} / 未标或跳过 {n_skip}）")
    L.append(f"- tokens：prompt {tok_in} / completion {tok_out} · 墙钟 {(time.time() - t0) / 60:.1f} 分钟")
    L.append(f"- 原始判定：`{raw_path.relative_to(REPO_ROOT)}`\n")
    L.append("> 口径：TPR=人工 fail 中裁判 fail 的比例；TNR=人工 pass 中裁判 pass 的比例；")
    L.append("> κ=Cohen's kappa（小样本 CI 宽，仅参照）；citation_accuracy 主表为 **composed**（语义 ∪ 代码层编号越界）。\n")

    for seg in segs_report:
        seg_label = "tuning+measurement（合并口径）" if seg == "all" else f"{seg}（{SEG_CN.get(seg, seg)}）"
        L.append(f"## {seg_label}\n")
        L.append("| criterion | n 对 | TPR（Wilson 95%） | TNR（Wilson 95%） | 一致率 | κ | 混淆 tp/fn/tn/fp |")
        L.append("|---|---|---|---|---|---|---|")
        for cr in criteria:
            m = report_m[seg][cr]
            c = m["confusion"]
            L.append(
                f"| {cr} | {m['n']} | {fmt_pct(m['tpr'])} | {fmt_pct(m['tnr'])} | "
                f"{fmt_pct(m['agreement'])} | {fmt_kappa(m['kappa'])} | "
                f"{c['tp']}/{c['fn']}/{c['tn']}/{c['fp']} |"
            )
        L.append("")

    # citation_accuracy 语义层 vs composed 对照（仅当本轮请求了该 criterion）
    if "citation_accuracy" in criteria:
        L.append("## citation_accuracy：judge 语义层 vs composed（代码层叠加）\n")
        L.append("| 段 | n 对 | 语义层 一致率 | composed 一致率 | 代码层判负数 |")
        L.append("|---|---|---|---|---|")
        for seg in segs_report:
            mj = judge_only_m[seg]["citation_accuracy"]
            mc = report_m[seg]["citation_accuracy"]
            n_code_fail = sum(
                1 for r in results
                if r["criterion"] == "citation_accuracy"
                and (seg == "all" or r["segment"] == seg)
                and r.get("code") and not r["code"]["pass"]
            )
            aj = fmt_pct(mj["agreement"])
            ac = fmt_pct(mc["agreement"])
            L.append(f"| {seg} | {mc['n']} | {aj} | {ac} | {n_code_fail} |")
        L.append("")

    # 分歧清单
    L.append("## 分歧清单（人工 vs composed）\n")
    n_disp = 0
    for r in results:
        h = (r.get("human") or {}).get("label")
        v = r.get("composed")
        if h in ("pass", "fail") and v in ("pass", "fail") and h != v:
            n_disp += 1
            jr = (r["judge"].get("reason") or "")[:110]
            hr = ((r.get("human") or {}).get("reason") or "")[:110]
            code_note = ""
            if r["criterion"] == "citation_accuracy" and r.get("code") and not r["code"]["pass"]:
                code_note = f" · 代码层越界编号 {r['code']['invalid_markers']}"
            L.append(f"- **{r['id']} · {r['criterion']}**：人工 **{h}**（{hr}）"
                     f" → 裁判 **{v}**（{jr}）{code_note}")
    if n_disp == 0:
        L.append("- （无分歧）")
    L.append("")

    report_path = reports_dir / f"calib_{args.segment}_{ts}.md"
    report_path.write_text("\n".join(L), encoding="utf-8")

    print(f"\n完成：{len(results)} 条判定（error {n_err}，缓存命中 {n_cached}）")
    for seg in segs_report:
        for cr in criteria:
            m = report_m[seg][cr]
            print(f"  [{seg}/{cr}] n={m['n']} 一致率={fmt_pct(m['agreement'])} "
                  f"κ={fmt_kappa(m['kappa'])} TPR={fmt_pct(m['tpr'])} TNR={fmt_pct(m['tnr'])}")
    print(f"\n原始判定: {raw_path}")
    print(f"报告: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
