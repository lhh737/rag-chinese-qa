"""第二裁判交叉检查（设计 §4.4）：对某个已完成的 judged run 抽样，用第二裁判重新判定，
与主裁判 verdict 对比，出具裁判间一致率与分歧清单（随运行报告披露）。

用法：
    uv run python scripts/judge_crosscheck.py --run-id 20261013-003 --sample 5
    # 抽样规则：按题号排序后每 K 条取 1（默认 5 ≈ 20%），确定性、可复现

输出：
    eval_results/<run_id>/crosscheck_<model>.md（报告）+ 控制台摘要
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ragqa.config.settings import get_settings  # noqa: E402
from ragqa.evaluation.datasets import load_jsonl  # noqa: E402
from ragqa.evaluation.judges.cache import JudgeCache  # noqa: E402
from ragqa.evaluation.judges.client import JudgeClient, JudgeError  # noqa: E402
from ragqa.evaluation.runner import criteria_for  # noqa: E402
from ragqa.models.factory import get_judge_client  # noqa: E402


def build_payload(item, answer: str, trace: dict) -> dict:
    ctx = trace.get("context") or {}
    texts = list(ctx.get("texts") or [])
    sources = list(ctx.get("sources") or [])
    sources += [""] * (len(texts) - len(sources))
    context_numbered = "\n\n".join(
        f"[{i}]（{s}）\n{t}" if s else f"[{i}]\n{t}"
        for i, (t, s) in enumerate(zip(texts, sources), start=1)
    )
    return {
        "question": item.question,
        "answer": answer,
        "context": "\n\n".join(texts),
        "context_numbered": context_numbered,
        "n_contexts": len(texts),
        "ground_truth": item.ground_truth or "",
        "key_claims": "\n".join(f"- {c}" for c in (item.key_claims or [])),
        "expected_behavior": item.expected_behavior,
        "injection_marker": getattr(item, "injection_marker", "") or "",
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--sample", type=int, default=5, help="每 K 条取 1（默认 5 ≈ 20%）")
    ap.add_argument("--model", default=None, help="第二裁判模型（缺省取 .env JUDGE_MODEL_SECOND）")
    ap.add_argument("--endpoint", default="official", choices=["official", "tokenhub"],
                    help="裁判端点：official=DeepSeek 官方；tokenhub=异族锚点（mimo 等）用")
    ap.add_argument("--prompt-version", default="v1")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    run_dir = REPO_ROOT / "eval_results" / args.run_id
    rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    traces = {json.loads(l)["trace_id"]: json.loads(l) for l in
              (run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    items = {i.id: i for i in load_jsonl(REPO_ROOT / "data/golden/golden_v1.jsonl")}
    items.update({i.id: i for i in load_jsonl(REPO_ROOT / "data/golden/safety_v1.jsonl")})

    s = get_settings()
    model = args.model or s.judge_model_second
    if not model:
        print("✗ 未配置第二裁判（--model 或 .env JUDGE_MODEL_SECOND）")
        return 2
    client2 = get_judge_client()
    if args.endpoint == "tokenhub":
        key = os.environ.get("TOKENHUB_API_KEY", "").strip()
        if not key:
            envp = REPO_ROOT / ".env"
            if envp.is_file():
                for line in envp.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("TOKENHUB_API_KEY="):
                        key = line.split("=", 1)[1].strip()
                        break
        if not key:
            print("✗ 未找到 TOKENHUB_API_KEY（环境变量或 .env）")
            return 2
        from openai import OpenAI
        client2 = OpenAI(api_key=key, base_url="https://tokenhub.tencentmaas.com/v1")
    judge2 = JudgeClient(model=model, cache=JudgeCache(), client=client2,
                         extra_body={}, max_tokens=args.max_tokens)

    sampled = [r for i, r in enumerate(sorted(rows, key=lambda x: x["id"])) if i % max(1, args.sample) == 0]
    tasks = []
    for r in sampled:
        item = items.get(r["id"])
        trace = traces.get(r.get("trace_id", "")) or {}
        if item is None or not trace:
            continue
        for cr, v in (r.get("judge") or {}).items():
            if (v or {}).get("verdict") in ("pass", "fail"):
                tasks.append((r, item, trace, cr, v["verdict"]))

    def run_one(t):
        r, item, trace, cr, main_v = t
        try:
            v2 = judge2.judge(item_id=r["id"], answer=(trace.get("generation") or {}).get("answer") or "",
                              criterion=cr, prompt_version=args.prompt_version,
                              payload=build_payload(item, (trace.get("generation") or {}).get("answer") or "", trace))
            return {"id": r["id"], "criterion": cr, "main": main_v, "second": v2.get("verdict"),
                    "second_reason": v2.get("reason", ""), "error": v2.get("error")}
        except JudgeError as e:
            return {"id": r["id"], "criterion": cr, "main": main_v, "second": "error", "second_reason": str(e)[:200]}

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        results = list(ex.map(run_one, tasks))

    valid = [x for x in results if x["second"] in ("pass", "fail")]
    agree = sum(1 for x in valid if x["main"] == x["second"])
    disagree = [x for x in valid if x["main"] != x["second"]]
    errors = [x for x in results if x["second"] == "error"]

    L = [f"# 第二裁判交叉检查 · {args.run_id} · {model}\n",
         f"- 抽样：每 {args.sample} 条取 1（题数 {len(sampled)}，判定对 {len(results)}）",
         f"- 裁判间一致率：{agree}/{len(valid)}" + (f" = {agree / len(valid):.3f}" if valid else "（无有效样本）"),
         f"- 第二裁判 error：{len(errors)}\n", "## 分歧清单\n"]
    for x in disagree:
        L.append(f"- **{x['id']} · {x['criterion']}**：主裁判 {x['main']} vs 第二裁判 {x['second']}（{x['second_reason'][:100]}）")
    if not disagree:
        L.append("- （无分歧）")
    out = run_dir / f"crosscheck_{model.replace('/', '_')}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"第二裁判（{model}）交叉：{agree}/{len(valid)} 一致，分歧 {len(disagree)}，error {len(errors)}")
    print(f"报告：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
