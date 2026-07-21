"""回答生成领域模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.retrieval.models import RetrievedChunk

if TYPE_CHECKING:
    from app.retrieval.context.packer import ContextCitation


@dataclass(frozen=True, slots=True)
class Citation:
    """生成回答中对用户可见的引用来源。"""

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

    @classmethod
    def from_context_citation(cls, citation: ContextCitation) -> "Citation":
        """把检索上下文的来源映射转换为最终回答引用。"""

        return cls(
            citation_id=citation.citation_id,
            chunk_id=citation.chunk_id,
            doc_id=citation.doc_id,
            version_id=citation.version_id,
            title=citation.title,
            source_path=citation.source_path,
            snippet=citation.snippet,
            page_start=citation.page_start,
            page_end=citation.page_end,
            section=citation.section,
        )


@dataclass(frozen=True, slots=True)
class RagAnswer:
    """一次 RAG 问答生成的最终结构化结果。"""

    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    trace_id: str
    latency_ms: float
