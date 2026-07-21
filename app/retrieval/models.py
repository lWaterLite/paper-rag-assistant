"""在线检索领域模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RetrievalSignal:
    """单个召回源对某个 chunk 提供的检索证据。"""

    retriever: str
    rank: int
    score: float


@dataclass(frozen=True, slots=True)
class RerankSignal:
    """重排序阶段对某个候选提供的运行时证据。"""

    reranker: str
    rank: int
    score: float


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """一次在线检索返回的结构化 chunk。"""

    chunk_id: str
    doc_id: str
    content_hash: str
    version_id: str
    text: str
    score: float
    rank: int
    retriever: str
    source_path: str
    chunk_index: int
    title: str | None = None
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieval_signals: tuple[RetrievalSignal, ...] = ()
    rerank_signal: RerankSignal | None = None
