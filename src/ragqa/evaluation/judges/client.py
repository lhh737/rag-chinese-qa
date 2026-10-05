"""Judge 客户端（设计 §4.4）：一个 criterion 一个 prompt，二分判定 + 证据必填。

- 输出严格 JSON：{"verdict": "pass|fail", "evidence": "原文摘录", "reason": "一句话"}
- temperature=0；JSON 解析失败 → 修复重试 1 次 → 仍失败抛 JudgeError（调用方记 judge_error，
  计入覆盖率申报、不计入分母）
- 缓存：键含 prompt 版本与模型；judge prompt / 模型变更即自然失效（防漂移靠版本化）
"""
from __future__ import annotations

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


class JudgeClient:
    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        cache: JudgeCache | None = None,
        client: Any | None = None,
        prompts_dir: str | Path | None = None,
    ) -> None:
        self.model = model or get_settings().gen_model
        self.temperature = temperature
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
        key = JudgeCache.make_key(
            item_id, answer, criterion, prompt_version, self.model, {"temperature": self.temperature}
        )
        cached = self.cache.get(key)
        if cached is not None:
            return {**cached, "criterion": criterion, "prompt_version": prompt_version, "_cached": True}

        template = self.load_prompt(criterion, prompt_version)
        try:
            prompt = template.format(**payload)
        except KeyError as e:
            raise JudgeError(f"judge prompt 模板缺字段 {e}（criterion={criterion}/{prompt_version}）") from e

        messages = [{"role": "user", "content": prompt}]
        t0 = time.time()
        last_err: Exception | None = None
        for attempt in range(2):  # 失败修复重试 1 次
            if attempt == 1:
                messages = messages + [
                    {"role": "assistant", "content": "<上一条输出无法解析为严格 JSON>"},
                    {"role": "user", "content": "请只输出一个 JSON 对象，不要任何其他文字或代码块标记。"},
                ]
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=600,
                    extra_body=thinking_off_extra_body(),
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
        return json.loads(m.group(0))
