"""自动归因引擎（taxonomy v1）：按 trace 信号对失败条目打类 + 证据（设计 §4.6/§4.10）。

与 `eval_results/analysis/error_analysis_v1.md` 的 6 类对齐；
规则为"证据优先"：每条归因附机器证据 dict，供人工抽检（查看器模式③）确认/纠正。

判定优先级（取第一个命中为 primary；其余命中记入 secondary）：
1. 无失败信号                          → 无问题
2. 唯一失败为引用（非法编号/张冠李戴） → citation_error
3. 误拒（规则拒答）且材料足（支撑已在最终上下文）→ generation_synthesis_failure
4. 对抗题 + 拒答/回避式作答           → premise_not_corrected
5. 支撑文档未进最终上下文：
   a. 支撑(全部或多跳的一边)从未进融合候选池 → retrieval_miss.not_recalled（含多跳单边）
   b. 支撑进了候选池但被截断线截掉          → retrieval_miss.ranked_below_cutoff
6. 其余失败（材料足、非拒答）          → omission（保留观察）
+ 辅助标记：hyde_used（供人工区分 query_rewrite_drift）、citation_invalid_markers 等。
"""
from __future__ import annotations

import re
from typing import Any

REFUSAL_PAT = ["无法回答", "资料未提及", "未提及", "未包含", "未提供", "无法从", "不能回答",
               "无法确定", "不足以回答", "资料不足以", "无法根据", "未找到相关", "没有相关"]
MARKER_RE = re.compile(r"\[(\d{1,3})\]|【(\d{1,3})】")


def _doc_of(chunk_uid: str) -> str:
    return str(chunk_uid).split(":")[0]


def fuse_doc_hits(trace: dict) -> list[str]:
    """融合候选池（文档级，按出现先后）。"""
    docs: list[str] = []
    for f in ((trace.get("rrf") or {}).get("fused") or []):
        d = _doc_of(f.get("chunk_uid", ""))
        if d and d not in docs:
            docs.append(d)
    return docs


def context_docs(trace: dict) -> list[str]:
    """最终送入模型的上下文（文档级）。"""
    docs: list[str] = []
    for pid in ((trace.get("context") or {}).get("parent_ids") or []):
        d = _doc_of(pid)
        if d and d not in docs:
            docs.append(d)
    return docs


def attribute_item(item, row: dict, trace: dict) -> dict:
    """返回 {primary, secondary[], evidence{...}, suggestion}（失败项）或 {'primary': '无问题'}。"""
    answer = (trace.get("generation") or {}).get("answer") or ""
    judge = row.get("judge") or {}
    failed_criteria = sorted(cr for cr, v in judge.items() if v.get("verdict") == "fail")
    if not failed_criteria:
        return {"primary": "无问题", "secondary": [], "evidence": {}}

    supporting = set(item.supporting_docs or [])
    fused = fuse_doc_hits(trace)
    ctx = context_docs(trace)
    fused_set, ctx_set = set(fused), set(ctx)
    refused = any(p in answer for p in REFUSAL_PAT)
    hyp_used = bool((trace.get("hyde") or {}).get("used"))

    # 引用代码层：越界编号（原文符号豁免）
    ctx_text = "\n".join((trace.get("context") or {}).get("texts") or [])
    invalid_markers = []
    for m in MARKER_RE.finditer(answer):
        n = int(m.group(1) or m.group(2))
        if not (1 <= n <= len((trace.get("context") or {}).get("texts") or [])) and m.group(0) not in ctx_text:
            invalid_markers.append(n)

    hit_ctx = sorted(supporting & ctx_set)
    miss_ctx = sorted(supporting - ctx_set)
    miss_all = sorted(supporting - fused_set)

    evidence: dict[str, Any] = {
        "failed_criteria": failed_criteria,
        "supporting_docs": sorted(supporting),
        "support_in_context": hit_ctx,
        "support_missing_from_context": miss_ctx,
        "support_never_retrieved": miss_all,
        "refused": refused,
        "hyde_used": hyp_used,
        "citation_invalid_markers": sorted(set(invalid_markers)),
    }

    secondary: list[str] = []
    primary: str
    ac_fail = judge.get("answer_correctness", {}).get("verdict") == "fail"
    cit_fail = judge.get("citation_accuracy", {}).get("verdict") == "fail"

    # 2. citation_error：仅引用失败（或引用失败 + 编号越界的代码实证）
    if failed_criteria == ["citation_accuracy"] or (cit_fail and invalid_markers and not ac_fail):
        primary = "citation_error"
    # 3. 误拒且材料足
    elif refused and ac_fail and supporting and not miss_ctx:
        primary = "generation_synthesis_failure"
    # 4. 对抗题拒答式回避
    elif (item.type == "adversarial") and ac_fail and refused:
        primary = "premise_not_corrected"
    # 5. 检索类
    elif supporting and miss_ctx:
        if miss_all:
            primary = "retrieval_miss.not_recalled"
            if item.type == "multi_hop" and (supporting & fused_set) and miss_all:
                primary = "retrieval_miss.not_recalled"  # 多跳单边
        else:
            primary = "retrieval_miss.ranked_below_cutoff"
        if hyp_used:
            secondary.append("hyde_used（人工区分 query_rewrite_drift）")
    # 6. 其余
    else:
        primary = "omission" if not refused else "generation_synthesis_failure"
        if refused and miss_ctx:
            primary = "retrieval_miss.ranked_below_cutoff" if (supporting & fused_set) else "retrieval_miss.not_recalled"

    return {"primary": primary, "secondary": secondary, "evidence": evidence}


_SUGGEST = {
    "retrieval_miss.not_recalled": "候选手段：切分/embedding/融合调参；多跳题关注第二来源召回（top-k 或按来源增补）",
    "retrieval_miss.ranked_below_cutoff": "候选手段：base_context_k 上调对照 / 多跳题按来源多样性增补",
    "query_rewrite_drift": "候选手段：HyDE prompt 约束（禁编具体文献/数字）/ 低置信回退原 query",
    "generation_synthesis_failure": "候选手段：prompt 多版本 A/B（'先综合再拒答'指令）",
    "premise_not_corrected": "候选手段：对抗前提处理指令",
    "citation_error": "候选手段：引用约束 prompt v2 + 输出后处理校验",
    "omission": "候选手段：上下文呈现/prompt（保留观察类）",
    "无问题": "",
}


def suggestion_of(primary: str) -> str:
    return _SUGGEST.get(primary, "")
