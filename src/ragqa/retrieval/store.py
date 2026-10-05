"""向量存储：faiss-cpu 直连（替代 langchain_community FAISS 包装，设计 §5.4）。

关键设计：
- `IndexFlatL2`：与旧 langchain FAISS 默认距离策略一致，保证依赖迁移对照可信；
  更换度量（如余弦/IP）属于"可比变更"，留给消融实验阶段。
- 行号映射：faiss 索引第 i 行 ↔ `child_documents[i]`，两者一起落盘；
  加载时做 count / dim 双校验，不一致即拒绝加载（宁可显式失败，不静默错配）。
- 持久化布局（faiss_db/）：
    index.faiss         faiss 原生索引
    child_docs.json     子块（顺序即索引行号）
    parent_store.json   父块 {parent_id: {page_content, metadata}}
    manifest.json       文档清单 {doc_id: {...}}
    index_meta.json     索引元信息（维度 / 嵌入模型 / 数量 / 时间 / 管道版本）
    tokenized_corpus.json  BM25 分词缓存（避免每次启动重新分词）
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from ragqa.config.settings import get_settings
from ragqa.types import Document
from ragqa.utils.logging import logger
from ragqa.utils.paths import resolve_path
from ragqa.utils.text import zh_tokenize

PIPELINE_VERSION = "p0-v1"


class StoreInconsistencyError(RuntimeError):
    pass


class VectorStore:
    def __init__(self, persist_dir: str | Path | None = None, embed_model: Any | None = None,
                 auto_load: bool = True) -> None:
        s = get_settings()
        self.persist_dir = Path(persist_dir) if persist_dir else s.faiss_dir
        self.persist_dir = resolve_path(self.persist_dir)
        self._embed = embed_model  # 懒取（测试可注入假嵌入）

        self.index: faiss.IndexFlatL2 | None = None
        self.child_documents: list[Document] = []
        self.tokenized_corpus: list[list[str]] = []
        self.parent_store: dict[str, dict[str, Any]] = {}
        self.manifest: dict[str, dict[str, Any]] = {}
        self.meta: dict[str, Any] = {}
        self._bm25: BM25Okapi | None = None

        if auto_load:
            self.load()

    # ── 路径 ─────────────────────────────────────────────

    @property
    def index_path(self) -> Path:
        return self.persist_dir / "index.faiss"

    @property
    def child_path(self) -> Path:
        return self.persist_dir / "child_docs.json"

    @property
    def parent_path(self) -> Path:
        return self.persist_dir / "parent_store.json"

    @property
    def manifest_path(self) -> Path:
        return self.persist_dir / "manifest.json"

    @property
    def meta_path(self) -> Path:
        return self.persist_dir / "index_meta.json"

    @property
    def tokens_path(self) -> Path:
        return self.persist_dir / "tokenized_corpus.json"

    @property
    def embed(self):
        if self._embed is None:
            from ragqa.models.factory import get_embed_model

            self._embed = get_embed_model()
        return self._embed

    # ── 属性 ─────────────────────────────────────────────

    @property
    def is_empty(self) -> bool:
        return self.index is None or len(self.child_documents) == 0

    @property
    def count(self) -> int:
        return len(self.child_documents)

    @property
    def file_list(self) -> list[dict]:
        return sorted(self.manifest.values(), key=lambda x: x.get("loaded_at", ""), reverse=True)

    # ── 加载 / 保存 ───────────────────────────────────────

    def load(self) -> None:
        if not self.persist_dir.is_dir():
            return
        try:
            if self.index_path.is_file():
                self.index = faiss.read_index(str(self.index_path))
            if self.child_path.is_file():
                raw = json.loads(self.child_path.read_text(encoding="utf-8"))
                self.child_documents = [Document(page_content=d["page_content"], metadata=d.get("metadata", {})) for d in raw]
            if self.parent_path.is_file():
                self.parent_store = json.loads(self.parent_path.read_text(encoding="utf-8"))
            if self.manifest_path.is_file():
                self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.meta_path.is_file():
                self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise StoreInconsistencyError(f"索引文件读取失败: {e}") from e

        # ── 一致性校验 ──
        if self.index is not None:
            if self.index.ntotal != len(self.child_documents):
                raise StoreInconsistencyError(
                    f"索引行数({self.index.ntotal}) 与子块数({len(self.child_documents)}) 不一致，"
                    f"请重建索引（ragqa ingest）"
                )
            expected_dim = int(self.meta.get("dim", 0) or 0)
            if expected_dim and self.index.d != expected_dim:
                raise StoreInconsistencyError(
                    f"索引维度({self.index.d}) 与元信息({expected_dim}) 不一致，请重建索引"
                )
            s = get_settings()
            if self.meta.get("embed_model") and self.meta["embed_model"] != s.embed_api_model and s.embed_mode == "api":
                logger.warning(
                    "[store] 索引由 %s 构建，当前配置为 %s —— 语义可能不一致，建议重建",
                    self.meta.get("embed_model"), s.embed_api_model,
                )

        # 分词缓存（count 一致时直接复用，否则重建）
        if self.child_documents:
            if self.tokens_path.is_file() and len(self.tokenized_corpus) == 0:
                try:
                    cached = json.loads(self.tokens_path.read_text(encoding="utf-8"))
                    if len(cached) == len(self.child_documents):
                        self.tokenized_corpus = cached
                except Exception:
                    logger.warning("[store] 分词缓存读取失败，将重建")
            if not self.tokenized_corpus:
                self.tokenized_corpus = [zh_tokenize(d.page_content) for d in self.child_documents]
                self._save_tokens()

    def save(self) -> None:
        """全量落盘。顺序：先子块（行号真相）后索引，任何中断都可被加载期校验发现。"""
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.child_path.write_text(
            json.dumps(
                [{"page_content": d.page_content, "metadata": d.metadata} for d in self.child_documents],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        self.parent_path.write_text(json.dumps(self.parent_store, ensure_ascii=False), encoding="utf-8")
        self.manifest_path.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        self.meta_path.write_text(json.dumps(self.meta, ensure_ascii=False, indent=1), encoding="utf-8")
        self._save_tokens()

    def _save_tokens(self) -> None:
        if self.tokenized_corpus:
            self.tokens_path.write_text(json.dumps(self.tokenized_corpus, ensure_ascii=False), encoding="utf-8")

    # ── 文档管理 ─────────────────────────────────────────

    def has_document(self, doc_id: str, content_hash: str | None = None) -> bool:
        info = self.manifest.get(doc_id)
        if not info:
            return False
        if content_hash is None:
            return True
        return info.get("content_hash") == content_hash

    def add_document(self, parents: list[Document], children: list[Document], doc_info: dict) -> bool:
        """新增一篇文档（父子块）。返回 False 表示已存在（幂等跳过）。"""
        doc_id = doc_info["doc_id"]
        if self.has_document(doc_id, doc_info.get("content_hash")):
            return False
        if self.has_document(doc_id):
            raise ValueError(f"doc_id 已存在但内容不同: {doc_id}（需先移除旧版，P7 快照机制处理）")

        # 父块入库
        for p in parents:
            pid = p.metadata["parent_id"]
            self.parent_store[pid] = {
                "page_content": p.page_content,
                "metadata": {k: v for k, v in p.metadata.items() if k != "parent_id"},
            }

        # 子块向量化 + 追加到索引
        if children:
            t0 = time.time()
            vectors = self.embed.embed_documents([c.page_content for c in children])
            arr = np.asarray(vectors, dtype="float32")
            if self.index is None:
                self.index = faiss.IndexFlatL2(arr.shape[1])
            if arr.shape[1] != self.index.d:
                raise ValueError(f"嵌入维度不一致: 新向量 {arr.shape[1]} vs 索引 {self.index.d}")
            self.index.add(arr)
            self.child_documents.extend(children)
            self.tokenized_corpus.extend(zh_tokenize(c.page_content) for c in children)
            self._bm25 = None  # 失效缓存
            logger.info("[store] %s: %d 子块向量化入库（%.1fs）", doc_id, len(children), time.time() - t0)

        doc_info = dict(doc_info)
        doc_info.update({"n_parents": len(parents), "n_children": len(children)})
        self.manifest[doc_id] = doc_info
        self.meta = {
            "dim": self.index.d if self.index is not None else None,
            "embed_model": get_settings().embed_api_model if get_settings().embed_mode == "api" else get_settings().embed_local_model,
            "count": len(self.child_documents),
            "pipeline_version": PIPELINE_VERSION,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        return True

    # ── 检索 ─────────────────────────────────────────────

    def search_with_scores(self, query: str, k: int) -> list[tuple[Document, float]]:
        """向量检索：返回 [(Document, L2 距离), ...]（距离越小越近）。"""
        if self.is_empty:
            return []
        qv = np.asarray([self.embed.embed_query(query)], dtype="float32")
        distances, indices = self.index.search(qv, min(k, self.index.ntotal))
        out: list[tuple[Document, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            out.append((self.child_documents[int(idx)], float(dist)))
        return out

    def bm25_rank(self, query: str, k: int) -> list[tuple[Document, float]]:
        """BM25 检索：返回 [(Document, 分数), ...]（分数越大越相关）。"""
        if not self.child_documents:
            return []
        if self._bm25 is None:
            self._bm25 = BM25Okapi(self.tokenized_corpus)
        q_tokens = zh_tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        # 稳定降序（与旧栈 sorted(..., reverse=True) 的并列语义一致：同分按语料顺序）
        top = np.argsort(-scores, kind="stable")[:k]
        return [(self.child_documents[int(i)], float(scores[int(i)])) for i in top]

    # ── 文件 ─────────────────────────────────────────────

    def get_parent(self, parent_id: str) -> dict[str, Any] | None:
        return self.parent_store.get(parent_id)

    def get_document_content(self, doc_id: str) -> str | None:
        info = self.manifest.get(doc_id)
        if not info:
            return None
        path = resolve_path(info.get("path", ""))
        if not path.is_file():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return None
