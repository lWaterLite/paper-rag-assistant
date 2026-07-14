"""尚未归入具体业务子系统的回答与检索模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievalSignal:
    """单个召回源对某个 chunk 提供的检索证据。"""

    retriever: str
    rank: int
    score: float


@dataclass(frozen=True)
class RerankSignal:
    """重排序阶段对某个候选提供的运行时证据。"""

    reranker: str
    rank: int
    score: float


@dataclass(frozen=True)
class RetrievedChunk:
    """检索结果。

    与 DocumentChunk 相比，它多了 score、rank、retriever 等查询时产生的信息。
    """

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


@dataclass(frozen=True)
class Citation:
    """回答中的引用来源。"""

    citation_id: str
    chunk_id: str
    doc_id: str
    version_id: str
    title: str | None
    source_path: str
    snippet: str
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None


@dataclass(frozen=True)
class RagAnswer:
    """一次 RAG 问答的最终结构化结果。"""

    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    trace_id: str
    latency_ms: float

