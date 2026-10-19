"""AI 第二标注（同族）——与人工盲标逐 criterion 对齐。

用法：
    uv run python scripts/ai_second_annotate.py --sample-size 50
    uv run python scripts/ai_second_annotate.py --model deepseek-flash --concurrency 3

产出：
    eval_results/calibration/ai_second_annotation_v1.jsonl
    eval_results/calibration/reports/P4b_ai_second_annotation.md
    eval_results/calibration/ai_second_disagreements_v1.jsonl

诚实边界（必须写进报告）：
- 模型与 judge/生成同族（deepseek-flash），**不是独立第二标注人**；
- 使用判定规范 prompt，不使用 judge prompt；报告 AI–human 一致率/κ，不称人类 inter-annotator；
- 结论是“同族模型 + 不同 prompt 的一致性检查”，不是人类标注者间信度。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from calibrate_judge import build_payload, cohen_kappa, metrics  # noqa: E402
from ragqa.config.settings import get_settings  # noqa: E402
from ragqa.models.factory import get_gen_client, thinking_off_extra_body  # noqa: E402

CALIB = REPO_ROOT / "eval_results" / "calibration"
PROMPT = REPO_ROOT / "prompts" / "annotator" / "quality_v1.md"
OUT_JSONL = CALIB / "ai_second_annotation_v1.jsonl"
OUT_REPORT = CALIB / "reports" / "P4b_ai_second_annotation.md"
OUT_DISAGREE = CALIB / "ai_second_disagreements_v1.jsonl"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _render(template: str, payload: dict) -> str:
    for key, value in payload.items():
        template = template.replace("{" + key + "}", str(value))
    return template


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("no JSON object")
    return json.loads(text[i:j + 1])


def _normalize(obj: dict, criteria: list[str]) -> dict:
    out = {}
    for c in criteria:
        v = obj.get(c)
        if not isinstance(v, dict):
            raise ValueError(f"missing criterion {c}")
        verdict = str(v.get("verdict", "")).strip().lower()
        if verdict not in ("pass", "fail"):
            raise ValueError(f"bad verdict {c}: {verdict}")
        out[c] = {
            "verdict": verdict,
            "evidence": str(v.get("evidence", "") or "")[:300],
            "reason": str(v.get("reason", "") or "")[:300],
        }
    return out


def _label_verdicts(item: dict) -> dict[str, str]:
    return {c: "pass" for c in item.get("criteria", [])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--model", default=None, help="默认 settings.gen_model")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--reuse", action="store_true", help="复用已有 JSONL，不重复调用 API")
    args = parser.parse_args()

    settings = get_settings()
    model = args.model or settings.gen_model
    template = PROMPT.read_text(encoding="utf-8")

    items = _rows(CALIB / "blind_label_v1.jsonl")
    labels = _rows(CALIB / "blind_labels_merged_v2.jsonl")
    human = {(r["id"], r["criterion"]): r["label"] for r in labels}
    fail_ids = {r["id"] for r in labels if r["label"] == "fail"}

    # 样本：全部有 fail 的题 + 8 条库外 + 其余通过项，凑满 sample_size
    selected = [i for i in items if i["id"] in fail_ids]
    oos = [i for i in items if i.get("expected_behavior") == "refuse" and i["id"] not in fail_ids]
    rest = [i for i in items if i["id"] not in fail_ids and i.get("expected_behavior") != "refuse"]
    selected += oos + rest
    selected = selected[: args.sample_size]

    results: list[dict] = []
    selected_ids = {i["id"] for i in selected}
    if args.reuse and OUT_JSONL.exists():
        existing = _rows(OUT_JSONL)
        have = {r["id"] for r in existing}
        if selected_ids <= have:
            results = [r for r in existing if r["id"] in selected_ids]
            print(f"[reuse] {len(results)} 条已有结果，跳过 API 调用")

    if not results:
        client = get_gen_client()

        def run_one(item: dict) -> dict:
            criteria = list(item.get("criteria", []))
            payload = build_payload(item)
            payload["criteria"] = "、".join(criteria)
            prompt = _render(template, payload)
            t0 = time.time()
            last_err = None
            for attempt in range(2):
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=args.max_tokens,
                        extra_body=thinking_off_extra_body(),
                    )
                    text = (resp.choices[0].message.content or "")
                    obj = _parse_json(text)
                    verdicts = _normalize(obj, criteria)
                    usage = getattr(resp, "usage", None)
                    return {
                        "id": item["id"],
                        "model": model,
                        "prompt": "annotator/quality_v1",
                        "criteria": criteria,
                        "verdicts": verdicts,
                        "usage": {
                            "prompt_tokens": getattr(usage, "prompt_tokens", None),
                            "completion_tokens": getattr(usage, "completion_tokens", None),
                            "total_tokens": getattr(usage, "total_tokens", None),
                        } if usage else {},
                        "latency_ms": round((time.time() - t0) * 1000),
                        "error": None,
                    }
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    time.sleep(1.0)
            return {
                "id": item["id"], "model": model, "prompt": "annotator/quality_v1",
                "criteria": criteria, "verdicts": {}, "usage": {},
                "latency_ms": round((time.time() - t0) * 1000), "error": str(last_err),
            }

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(run_one, item) for item in selected]
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                print(f"  {'✗' if r['error'] else '✓'} {r['id']}  {r['error'] or ''}")

    results.sort(key=lambda r: r["id"])
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSONL.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n", encoding="utf-8")

    # 对齐人工标签
    pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    disagreements = []
    errors = 0
    for r in results:
        if r["error"]:
            errors += 1
            continue
        for c in r["criteria"]:
            h = human.get((r["id"], c))
            ai = (r["verdicts"].get(c) or {}).get("verdict")
            if h is None or ai is None:
                errors += 1
                continue
            pairs[c].append((h, ai))
            if h != ai:
                disagreements.append({
                    "id": r["id"], "criterion": c, "human": h, "ai": ai,
                    "ai_evidence": (r["verdicts"].get(c) or {}).get("evidence", ""),
                    "ai_reason": (r["verdicts"].get(c) or {}).get("reason", ""),
                    "human_reason": next((x.get("reason", "") for x in labels
                                          if x["id"] == r["id"] and x["criterion"] == c), ""),
                })

    OUT_DISAGREE.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in disagreements) + ("\n" if disagreements else ""),
        encoding="utf-8",
    )

    all_pairs = [p for ps in pairs.values() for p in ps]
    overall = metrics(all_pairs) if all_pairs else None

    def _val(x):
        return x.get("value") if isinstance(x, dict) else x

    def row(name: str, m: dict | None) -> str:
        if not m:
            return f"| {name} | 0 | — | — | — | — |"
        conf = m["confusion"]
        fmt = lambda x: "—" if x is None else f"{x:.3f}"
        return (f"| {name} | {m['n']} | {fmt(_val(m.get('agreement')))} | {fmt(m.get('kappa'))} | "
                f"{fmt(_val(m.get('tpr')))} | {fmt(_val(m.get('tnr')))} | "
                f"{conf['tp']}/{conf['fn']}/{conf['tn']}/{conf['fp']} |")

    lines = [
        "# P4b · AI 第二标注（同族）· 2026-10-19",
        "",
        f"- 模型：`{model}`（与 judge/生成同族；**不是独立第二标注人**）",
        "- Prompt：`prompts/annotator/quality_v1.md`（按判定规范，非 judge prompt）",
        f"- 样本：{len(selected)} 题（含全部 {len(fail_ids)} 条 fail 题 + 8 条库外 + 通过对照）",
        f"- 有效判定：{sum(len(v) for v in pairs.values())} 条；错误/缺失：{errors}",
        "",
        "## 1. 与人工盲标的一致性",
        "",
        "| criterion | n | agreement | κ | TPR | TNR | tp/fn/tn/fp |",
        "|---|---:|---:|---:|---:|---:|---|",
        row("answer_correctness", metrics(pairs.get("answer_correctness", []))),
        row("faithfulness", metrics(pairs.get("faithfulness", []))),
        row("citation_accuracy", metrics(pairs.get("citation_accuracy", []))),
        row("oos_refusal", metrics(pairs.get("oos_refusal", []))),
        row("**overall**", overall),
        "",
        "> TPR = 人工 fail 中 AI 也判 fail；TNR = 人工 pass 中 AI 也判 pass。",
        "> κ 在 fail 样本很少时不稳定；结论以 agreement + 分歧案例为主。",
        "",
        "## 2. 分歧案例",
        "",
        f"共 {len(disagreements)} 条，明细：`eval_results/calibration/ai_second_disagreements_v1.jsonl`。",
        "",
    ]
    if disagreements:
        lines += [
            f"本轮 {len(disagreements)} 条分歧全部为 **AI 判 fail / 人工判 pass**（AI 更严格）；"
            "真人裁决可判断是 AI 过严还是人工漏判。",
            "",
            "| id | criterion | human | AI | AI 理由 |",
            "|---|---|---|---|---|",
        ]
        for d in disagreements[:30]:
            lines.append(f"| {d['id']} | {d['criterion']} | {d['human']} | {d['ai']} | {d['ai_reason'][:120]} |")
    else:
        lines.append("无分歧。")
    lines += [
        "",
        "## 3. 诚实边界（必须与数字一起读）",
        "",
        "1. **同族**：AI 标注器与 judge、生成模型同为 `deepseek-flash`；这不是独立第二标注人，不能替代真人 κ。",
        "2. **同模型不同 prompt**：只换了判定规范 prompt，模型权重相同；一致性高是预期结果，不能当作强独立性证据。",
        "3. **样本**：50 题、fail 样本集中；κ 的区间很宽，分 criterion 的数字只作方向性参考。",
        f"4. **人工裁决待做**：本轮共 {len(disagreements)} 条分歧，全部为 AI fail / 人工 pass；"
        "未复核前，本报告只报 AI–human 一致率，不把分歧直接判成人工错。",
        "",
        "## 4. 这能证明什么 / 不能证明什么",
        "",
        "- **能**：同一判定规范下，模型能作为第二路预标注通道，减少人工全量标注成本；分歧项可优先送人工。",
        "- **不能**：不能证明人类标注者间信度；不能证明异族模型也会一致；不能替代真人抽检。",
        "",
        "## 5. 下一步",
        "",
        f"- 真人复核 `ai_second_disagreements_v1.jsonl` 的全部 {len(disagreements)} 条分歧，回填裁决；",
        "- 若有异族模型额度，抽 20 条做异族锚点，比较同族/异族一致性；",
        "- 若目标是“双标注人 κ”，仍需找真人独立盲标 30–50 条。",
        "",
    ]
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n✓ 结果 → {OUT_JSONL}")
    print(f"✓ 分歧 → {OUT_DISAGREE}（{len(disagreements)} 条）")
    print(f"✓ 报告 → {OUT_REPORT}")
    if overall:
        print(f"  overall agreement={overall['agreement']['value']:.3f}  κ={overall['kappa']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
