"""Judge 结果缓存（设计 §4.4）：SQLite 单文件、零运维。

缓存键 = sha256(题号 + 答案hash + criterion + judge prompt 版本 + judge 模型 + 参数)。
命中即复用：降本 + 支撑 CI 离线回放（回放时全部命中缓存）。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

from ragqa.utils.paths import REPO_ROOT, resolve_path

DEFAULT_PATH = REPO_ROOT / "cache" / "judge_cache.db"


class JudgeCache:
    def __init__(self, path: str | Path = DEFAULT_PATH) -> None:
        self.path = resolve_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS judge_cache "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        self._conn.commit()

    @staticmethod
    def make_key(item_id: str, answer: str, criterion: str, prompt_version: str, model: str, params: dict | None = None) -> str:
        base = json.dumps(
            {
                "item_id": item_id,
                "answer_sha": hashlib.sha256((answer or "").encode("utf-8")).hexdigest(),
                "criterion": criterion,
                "prompt_version": prompt_version,
                "model": model,
                "params": params or {},
            },
            ensure_ascii=False, sort_keys=True,
        )
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM judge_cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO judge_cache (key, value, created_at) VALUES (?, ?, ?)",
                (key, json.dumps(value, ensure_ascii=False), time.strftime("%Y-%m-%dT%H:%M:%S")),
            )
            self._conn.commit()

    def stats(self) -> dict:
        with self._lock:
            n = self._conn.execute("SELECT COUNT(*) FROM judge_cache").fetchone()[0]
        return {"n_entries": n, "path": str(self.path)}
