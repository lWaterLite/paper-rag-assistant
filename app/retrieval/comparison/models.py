"""检索比较模式的领域模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from app.retrieval.models import RetrievedChunk
from app.core.tracing import RagTrace
from app.retrieval.configuration.retrieval import RetrievalStrategy


ComparisonStatus = Literal["success", "partial_error", "error"]
ComparedStrategyStatus = Literal["success", "error"]


@dataclass(frozen=True, slots=True)
class ComparedStrategyResult:
    """某一个检索策略的独立执行结果。"""

    retriever: RetrievalStrategy
    status: ComparedStrategyStatus
    results: tuple[RetrievedChunk, ...] = ()
    trace: RagTrace | None = None
    report_path: Path | None = None
    error_code: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class ComparedChunkOverlap:
    """多个策略共同命中的 chunk 摘要。"""

    chunk_id: str
    retrievers: tuple[RetrievalStrategy, ...]
    ranks_by_retriever: dict[RetrievalStrategy, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievalComparisonResult:
    """一次 compare search 的领域层结果。"""

    query: str
    top_k: int
    retrievers: tuple[RetrievalStrategy, ...]
    status: ComparisonStatus
    strategy_results: tuple[ComparedStrategyResult, ...]
    overlaps: tuple[ComparedChunkOverlap, ...]
    trace: RagTrace
    report_path: Path | None = None
