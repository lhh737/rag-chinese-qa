"""依赖守卫测试：确保代码库不再依赖 langchain（防回归，设计 §5.4 迁移纪律）。"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import ragqa

# 只匹配 import 语句（注释/文档里解释迁移历史是允许的）
_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+langchain\w*", re.MULTILINE)


def test_no_langchain_imports_in_source():
    """源码中不得出现 langchain 的 import（含延迟导入）。"""
    root = Path(ragqa.__file__).resolve().parent
    offenders = [
        str(p.relative_to(root))
        for p in root.rglob("*.py")
        if _IMPORT_RE.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == [], f"发现 langchain import: {offenders}"


def test_no_langchain_modules_loaded():
    """导入核心模块后，进程中不应出现任何 langchain 模块。"""
    for mod in ["ragqa.core", "ragqa.cli", "ragqa.ingestion.pipeline", "ragqa.retrieval.hybrid", "ragqa.generation.generator"]:
        importlib.import_module(mod)
    loaded = [m for m in sys.modules if "langchain" in m]
    assert loaded == [], f"launched langchain modules: {loaded}"
