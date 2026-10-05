"""文档清洗（设计 §4.13 防御第 1 层）。

- 剥离零宽字符 / 控制符（记录数量，供语料体检与迁移对照审计）
- 标记可疑指令样文本（中英文模式）：只标记与计数，**不静默删除**（保留原文可追溯）
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# 零宽字符 / 双向控制符 / C0 控制符 / 孤立代理项（保留 \t \n \r）——用码点判断，源码中不出现不可见字符
# 孤立代理项（U+D800–DFFF）来自 PDF 数学字体抽取（如 𝕏 被抽成半个代理对），
# 属非法 Unicode 标量值，会令 UTF-8 编码与 JSON 落盘失败——必须在清洗层剥离（P0 实弹发现）。
_STRIP_RANGES: tuple[tuple[int, int], ...] = (
    (0x200B, 0x200F),  # 零宽空格 / 连接符 / 方向标记
    (0x202A, 0x202E),  # 双向控制符
    (0x2060, 0x2060),  # word joiner
    (0xFEFF, 0xFEFF),  # BOM / 零宽不换行空格
    (0x0000, 0x0008),  # C0 控制符（保留 \t\n\r）
    (0x000B, 0x000C),
    (0x000E, 0x001F),
    (0xD800, 0xDFFF),  # 孤立代理项（非法，非合法字符对）
)

# 可疑指令模式（中英文，大小写不敏感）；命中只告警，不改内容
_SUSPICIOUS_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("ignore_instructions", re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)", re.I)),
    ("disregard_instructions", re.compile(r"disregard\s+(all\s+)?(previous|prior|above)", re.I)),
    ("role_override_en", re.compile(r"you\s+are\s+now\s+(a|an|the)\b", re.I)),
    ("ignore_zh", re.compile(r"忽略(以上|之前|前面|所有)(的)?(所有)?(指令|指示|提示|规则)")),
    ("disregard_zh", re.compile(r"无视(以上|之前|前面|所有)")),
    ("new_instruction_zh", re.compile(r"新的?(系统)?指令\s*[:：]")),
    ("system_prompt", re.compile(r"system\s*prompt", re.I)),
    ("jailbreak_zh", re.compile(r"越狱|开发者模式")),
]


@dataclass
class SanitizeStats:
    chars_removed: int = 0
    suspicious: list[str] = field(default_factory=list)  # 命中的模式名（去重）

    def merge(self, other: "SanitizeStats") -> None:
        self.chars_removed += other.chars_removed
        for name in other.suspicious:
            if name not in self.suspicious:
                self.suspicious.append(name)


def _is_stripped(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _STRIP_RANGES)


def strip_dirty_chars(text: str) -> tuple[str, int]:
    """剥离零宽/控制符，返回 (清洗后文本, 剥离数量)。"""
    removed = sum(1 for ch in text if _is_stripped(ch))
    if not removed:
        return text, 0
    return "".join(ch for ch in text if not _is_stripped(ch)), removed


def scan_suspicious(text: str) -> list[str]:
    return [name for name, pat in _SUSPICIOUS_PATTERNS if pat.search(text)]


def sanitize_text(text: str) -> tuple[str, SanitizeStats]:
    stats = SanitizeStats(suspicious=scan_suspicious(text))
    cleaned, stats.chars_removed = strip_dirty_chars(text)
    return cleaned, stats
