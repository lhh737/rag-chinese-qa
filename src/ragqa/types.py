"""核心数据类型：自有 dataclass（替代 langchain Document），全链路共享。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """文本块（与 langchain Document 字段兼容，便于迁移期对照）。"""

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HydeResult:
    enabled: bool
    used: bool                      # 实际是否使用了 HyDE 输出（失败回退时 False）
    query: str                      # 实际用于检索的 query
    output: str | None = None
    error: str | None = None
    usage: dict[str, int] | None = None
    latency_ms: float = 0.0


@dataclass
class ChunkHit:
    """单路召回的一次命中。"""

    chunk_uid: str
    score: float                    # 向量=距离（越小越近）；BM25=分数（越大越好）
    rank: int                       # 1 起
    parent_id: str | None = None
    doc_id: str | None = None


@dataclass
class ContextBlock:
    """最终送入生成的父块（rank 与答案中的 [n] 对应）。"""

    rank: int
    parent_id: str
    doc_id: str
    source: str
    text: str
    rerank_score: float | None = None


@dataclass
class Citation:
    """逐句引用：答案句 ↔ [n] 标记 ↔ 被引上下文。"""

    answer_sentence: str
    marker: int
    valid: bool
    source_doc: str | None = None
    parent_id: str | None = None
    supporting_snippet: str | None = None


@dataclass
class GenerationResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    citation_stats: dict[str, int] = field(default_factory=dict)
    model: str = ""
    prompt_version: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass
class RetrievalTrace:
    vector_hits: list[ChunkHit] = field(default_factory=list)
    bm25_hits: list[ChunkHit] = field(default_factory=list)
    fusion: dict[str, Any] = field(default_factory=dict)
    rerank: dict[str, Any] = field(default_factory=dict)
    context_parent_ids: list[str] = field(default_factory=list)
    context_chars: int = 0
    latency_ms: float = 0.0


@dataclass
class RetrievalResult:
    blocks: list[ContextBlock]
    trace: RetrievalTrace
    context: str = ""               # 格式化后的上下文（【参考资料i】…，[n] 引用与 block.rank 对应）


@dataclass
class QueryResult:
    answer: str
    contexts: list[ContextBlock]
    citations: list[Citation]
    trace: dict[str, Any]           # Trace Schema v1（dict，可直接落盘 JSONL）


@dataclass
class StreamChunk:
    """astream 的流式块；最后一个块 done=True 并携带完整结果。"""

    delta: str = ""
    done: bool = False
    result: QueryResult | None = None
