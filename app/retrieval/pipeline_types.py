"""Retrieval pipeline 共享运行时模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.retrieval.models import RetrievedChunk
from app.retrieval.configuration.retrieval import RetrievalStrategy


@dataclass(frozen=True, slots=True)
class RetrievalPipelineContext:
    """一次 retrieval pipeline 的已解析运行时参数。"""

    query: str
    retriever: RetrievalStrategy
    candidate_limit: int
    top_k: int


@dataclass(frozen=True, slots=True)
class RetrievalStageResult:
    """检索后处理阶段的输出及其可观测摘要。"""

    chunks: list[RetrievedChunk]
    detail: dict[str, Any] = field(default_factory=dict)
