"""Prompt 版本注册表：prompts/answer/vN.md（评测 CLI 可指定版本，变更进 Registry）。

模板占位符：{context}（参考资料文本）。用户问题通过 user 消息传入。
"""
from __future__ import annotations

from ragqa.utils.paths import REPO_ROOT

PROMPTS_DIR = REPO_ROOT / "prompts"


def load_answer_prompt(version: str) -> str:
    path = PROMPTS_DIR / "answer" / f"{version}.md"
    if not path.is_file():
        raise FileNotFoundError(f"answer prompt 版本不存在: {path}")
    return path.read_text(encoding="utf-8")


def list_answer_prompt_versions() -> list[str]:
    d = PROMPTS_DIR / "answer"
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("v*.md"))
