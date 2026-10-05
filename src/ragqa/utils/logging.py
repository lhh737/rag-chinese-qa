"""统一日志：控制台 + 按日切分文件（UTF-8）。"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from ragqa.utils.paths import REPO_ROOT, resolve_path

LOG_ROOT: Path = resolve_path(os.getenv("LOG_DIR", str(REPO_ROOT / "logs")))
LOG_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_LOG_FORMAT = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)


def get_logger(
    name: str = "rag",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_file: str | None = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:  # 避免重复添加 Handler
        return logger

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(console_handler)

    if not log_file:
        log_file = str(LOG_ROOT / f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(file_handler)

    return logger


logger = get_logger()
