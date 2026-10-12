# Judge Prompt 注册表（设计 §4.4）

约定：一个 criterion 一个目录，`prompts/judge/{criterion}/vN.md`；版本号进 Registry，
变更即"防漂移"回归（重跑测量段校准）。

每个 prompt 需包含：失败模式定义 / pass-fail 边界规则 / 2-4 个 few-shot（仅取自校准集"示例段"）/
输出格式（严格 JSON）。

输出格式（全 criterion 统一）：

```json
{"verdict": "pass|fail", "evidence": "上下文/答案原文摘录", "reason": "一句话"}
```

模板字段（`{field}` 按字段名替换——正文含 JSON 示例的花括号，故不走 `str.format`；见
`judges/client.py::_render_prompt`）。由调用方传入 payload 决定，当前通用字段：
`{question}` `{answer}` `{context}` `{context_numbered}` `{n_contexts}` `{ground_truth}`
`{key_claims}` `{expected_behavior}` `{injection_marker}`。

**状态（2026-10-12）**：P4 各 criterion prompt **v1 已撰写**（边界规则对齐
`docs/annotation-guideline.md` v1 + 错误分析 taxonomy v1；质量四件套的 few-shot 取自校准集
"示例段"，安全三件套 few-shot 待安全轻量校准时补充）。校准运行器：
`scripts/calibrate_judge.py`（调参段 → 测量段纪律内置）。目录：

- `answer_correctness/vN.md` —— 对照 ground_truth + key_claims 的要点完备性
- `faithfulness/vN.md` —— 每个可核验陈述均被上下文支撑
- `citation_accuracy/vN.md` —— 被引陈述确被所引上下文支撑（judge 部分；代码部分见 citations.py）
- `oos_refusal/vN.md` —— 库外题的正确拒答
- `injection_resistance/vN.md` —— 注入题"语义服从"判定（跨块变体必需）
- `harmful_refusal/vN.md` —— 有害请求拒绝且不含操作性内容
- `boundary_compliance/vN.md` —— 不泄系统提示词/不越权表态
