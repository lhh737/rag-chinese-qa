"""Judge 客户端（设计 §4.4）：一个 criterion 一个 prompt，二分判定 + 证据必填。

- 输出严格 JSON：{"verdict": "pass|fail", "evidence": "原文摘录", "reason": "一句话"}
- temperature=0；JSON 解析失败 → 修复重试 1 次 → 仍失败抛 JudgeError（调用方记 judge_error，
  计入覆盖率申报、不计入分母）
- 缓存：键含 prompt 版本+**内容哈希**、模型与推理参数；prompt / 模型变更即自然失效（防漂移）
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from ragqa.config.settings import get_settings
from ragqa.evaluation.judges.cache import JudgeCache
from ragqa.models.factory import get_gen_client, thinking_off_extra_body
from ragqa.utils.llm import usage_dict
from ragqa.utils.logging import logger
from ragqa.utils.paths import REPO_ROOT

JUDGE_PROMPTS_DIR = REPO_ROOT / "prompts" / "judge"


class JudgeError(RuntimeError):
    pass


_PLACEHOLDER = re.compile(r"\{([a-z_][a-z0-9_]*)\}")


def _render_prompt(template: str, payload: dict[str, Any]) -> str:
    """按字段名替换 {field} 占位符（单遍替换）。

    不用 str.format：judge prompt 正文含 JSON 示例（花括号），format 会误解析。
    单遍替换保证**载荷原文中出现的 {word} 字面量**（如检索到的含模板文本的文档）
    既不会被误判为缺字段，也不会被二次替换（2026-10-13 q053 实测修复）。
    """
    missing = sorted({m.group(1) for m in _PLACEHOLDER.finditer(template)} - set(payload))
    if missing:
        raise JudgeError(f"judge prompt 模板引用了未提供字段 {missing}")
    return _PLACEHOLDER.sub(lambda m: str(payload[m.group(1)]) if m.group(1) in payload else m.group(0), template)


class JudgeClient:
    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        cache: JudgeCache | None = None,
        client: Any | None = None,
        prompts_dir: str | Path | None = None,
        extra_body: dict | None = None,
        max_tokens: int = 600,
    ) -> None:
        self.model = model or get_settings().gen_model
        self.temperature = temperature
        # 供应商差异（2026-10 实测）：DeepSeek/DashScope 可关思考；TokenHub 的
        # 思考型裁判（kimi/minimax）不接受关闭参数（400）→ 显式传 {} 表示原样请求。
        self.extra_body = thinking_off_extra_body() if extra_body is None else extra_body
        self.max_tokens = max_tokens
        self.cache = cache if cache is not None else JudgeCache()
        self._client = client
        self.prompts_dir = Path(prompts_dir) if prompts_dir else JUDGE_PROMPTS_DIR

    @property
    def client(self):
        if self._client is None:
            self._client = get_gen_client()
        return self._client

    # ── prompt 加载 ──────────────────────────────────────

    def load_prompt(self, criterion: str, version: str) -> str:
        path = self.prompts_dir / criterion / f"{version}.md"
        if not path.is_file():
            raise JudgeError(f"judge prompt 不存在: {path}（P4 阶段定稿各 criterion 的 prompt）")
        return path.read_text(encoding="utf-8")

    # ── 判定 ─────────────────────────────────────────────

    def judge(
        self,
        *,
        item_id: str,
        answer: str,
        criterion: str,
        prompt_version: str,
        payload: dict[str, Any],
    ) -> dict:
        """返回 {verdict, evidence, reason, criterion, prompt_version, usage?, latency_ms?, _cached?}。"""
        template = self.load_prompt(criterion, prompt_version)
        # 缓存键含 prompt 内容哈希：调参时原地修改 vN.md 也能使旧判定自然失效（防静默漂移）
        prompt_sha = hashlib.sha256(template.encode("utf-8")).hexdigest()[:12]
        key = JudgeCache.make_key(
            item_id, answer, criterion, f"{prompt_version}:{prompt_sha}", self.model,
            {"temperature": self.temperature, "max_tokens": self.max_tokens, "extra_body": self.extra_body},
        )
        cached = self.cache.get(key)
        if cached is not None:
            return {**cached, "criterion": criterion, "prompt_version": prompt_version, "_cached": True}

        prompt = _render_prompt(template, payload)

        messages = [{"role": "user", "content": prompt}]
        t0 = time.time()
        last_err: Exception | None = None
        for attempt in range(2):  # 失败修复重试 1 次
            if attempt == 1:
                messages = messages + [
                    {"role": "assistant", "content": "<上一条输出无法解析为严格 JSON>"},
                    {"role": "user", "content": "请只输出一个 JSON 对象，不要任何其他文字或代码块标记；"
                                                "verdict、evidence、reason 三个字段必填，"
                                                "evidence 必须引用原文摘录（不得为空；无引文可摘时填写判定依据摘要），"
                                                "evidence 摘录保持简短（合计 60 字以内）。"},
                ]
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    # 思考型裁判可能把预算烧在 reasoning/长摘录上导致空输出或截断——
                    # 重试时大幅放宽预算（2026-10-13 q014 空输出 / q011 截断实测）
                    max_tokens=self.max_tokens if attempt == 0 else self.max_tokens * 4,
                    extra_body=self.extra_body,
                )
                content = (resp.choices[0].message.content or "").strip()
                data = self._parse_json(content)
                verdict = str(data.get("verdict", "")).lower()
                if verdict not in ("pass", "fail"):
                    raise ValueError(f"verdict 非法: {verdict!r}")
                evidence = str(data.get("evidence", "")).strip()
                if not evidence:
                    raise ValueError("evidence 缺失（无引文的判定视为可疑）")
                result = {
                    "verdict": verdict,
                    "evidence": evidence,
                    "reason": str(data.get("reason", "")).strip(),
                    "usage": usage_dict(resp),
                    "latency_ms": round((time.time() - t0) * 1000, 1),
                }
                self.cache.set(key, {k: result[k] for k in ("verdict", "evidence", "reason")})
                return {**result, "criterion": criterion, "prompt_version": prompt_version}
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning("[judge] %s/%s 第 %d 次失败: %s", criterion, item_id, attempt + 1, str(e)[:120])
        raise JudgeError(f"judge 判定失败（{criterion}/{item_id}）: {last_err}")

    @staticmethod
    def _parse_json(content: str) -> dict:
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            raise ValueError(f"无 JSON: {content[:120]}")
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
        # 兜底：输出里可能夹带多段 JSON/说明文字——从每个 '{' 起做括号配平，
        # 取第一个可解析且含 verdict 的对象（2026-10-13 q007 实测）。
        for start in (i for i, ch in enumerate(content) if ch == "{"):
            depth = 0
            in_str = esc = False
            for i in range(start, len(content)):
                ch = content[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            data = json.loads(content[start : i + 1])
                        except json.JSONDecodeError:
                            break
                        if isinstance(data, dict) and "verdict" in data:
                            return data
                        break
        raise ValueError(f"无法解析 JSON: {content[:120]}")
