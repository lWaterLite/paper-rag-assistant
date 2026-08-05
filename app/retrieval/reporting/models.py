"""Retrieval 报告领域模型。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from app.core.tracing import RagTrace
from app.retrieval.configuration.postprocessing.profile import PostProcessingProfile
from app.retrieval.models import RetrievedChunk


@dataclass(frozen=True, slots=True)
class RetrievalIndexSnapshot:
    """一次检索所依赖的索引身份与核心构建信息。"""

    index_id: str
    schema_version: int
    status: str
    artifact_definition_hash: str
    document_set_hash: str
    document_count: int
    chunk_count: int
    vector_count: int
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    vector_repository_type: str
    vector_collection_name: str
    distance_metric: str


@dataclass(frozen=True, slots=True)
class RetrievalConfigSnapshot:
    """一次检索使用的可审查配置快照。"""

    default_strategy: str
    default_top_k: int
    deduplicate_by_chunk_id: bool
    tokenizer_strategy: str
    bm25_k1: float
    bm25_b: float
    hybrid_candidate_multiplier: int
    hybrid_rrf_rank_constant: int
    hybrid_vector_weight: float
    hybrid_bm25_weight: float
    postprocessing: PostProcessingProfile
    registered_strategies: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalRuntimeSnapshot:
    """索引身份与 retrieval 配置组成的运行时快照。"""

    index: RetrievalIndexSnapshot
    config: RetrievalConfigSnapshot


@dataclass(frozen=True, slots=True)
class RetrievalStageObservation:
    """Retrieval 内部某个处理阶段的输入、输出和耗时。"""

    stage: str
    status: Literal["success", "error"]
    input_count: int
    output_count: int
    latency_ms: float
    detail: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RetrievalExecutionReport:
    """一次 retrieval 执行的完整报告数据。"""

    query: str
    requested_top_k: int | None
    resolved_top_k: int | None
    resolved_candidate_limit: int | None
    requested_retriever: str | None
    resolved_retriever: str | None
    candidate_count: int
    deduplicated_count: int
    returned_count: int
    stage_observations: tuple[RetrievalStageObservation, ...]
    results: tuple[RetrievedChunk, ...]
    runtime: RetrievalRuntimeSnapshot
    trace: RagTrace
    error_code: str | None = None
    error_message: str | None = None
    generated_at: str = ""

    def __post_init__(self) -> None:
        if not self.generated_at:
            object.__setattr__(self, "generated_at", datetime.now(UTC).isoformat())


@dataclass(frozen=True, slots=True)
class RetrievalComparisonStrategyReport:
    """聚合报告中单个策略的执行摘要。"""

    retriever: str
    status: str
    returned_count: int
    child_trace_id: str | None
    child_trace_status: str | None
    child_latency_ms: float | None
    report_path: str | None
    error_code: str | None
    error_message: str | None


@dataclass(frozen=True, slots=True)
class RetrievalComparisonOverlapReport:
    """聚合报告中多个策略共同命中的 chunk 摘要。"""

    chunk_id: str
    retrievers: tuple[str, ...]
    ranks_by_retriever: dict[str, int]


@dataclass(frozen=True, slots=True)
class RetrievalComparisonExecutionReport:
    """一次 compare search 的完整聚合报告数据。"""

    query: str
    top_k: int
    retrievers: tuple[str, ...]
    status: str
    strategy_results: tuple[RetrievalComparisonStrategyReport, ...]
    overlaps: tuple[RetrievalComparisonOverlapReport, ...]
    runtime: RetrievalRuntimeSnapshot
    trace: RagTrace
    generated_at: str = ""

    def __post_init__(self) -> None:
        if not self.generated_at:
            object.__setattr__(self, "generated_at", datetime.now(UTC).isoformat())
