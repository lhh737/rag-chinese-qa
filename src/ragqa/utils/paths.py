"""仓库路径工具：所有路径锚定仓库根，不依赖当前工作目录。"""
from __future__ import annotations

from pathlib import Path

# src/ragqa/utils/paths.py -> repo/
REPO_ROOT: Path = Path(__file__).resolve().parents[3]


def resolve_path(p: str | Path) -> Path:
    """相对路径按仓库根解析；绝对路径原样返回。"""
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path)
