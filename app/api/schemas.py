"""API 请求与响应 schema。

这里使用 Pydantic model 描述外部 API 边界，但暂时不引入 FastAPI。
这样可以先把 JSON 契约、字段校验和领域模型映射稳定下来，后续接入 Web 框架时直接复用。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.models import Citation, RagAnswer, RagTrace, RetrievedChunk
from app.retrieval.comparison import RetrievalComparisonResult
from app.retrieval.configs import RetrievalStrategy


class ApiModel(BaseModel):
    """所有 API schema 的基础模型。

    extra="forbid" 可以尽早发现调用方传错字段的问题，避免静默吞掉错误输入。
    """

    model_config = ConfigDict(extra="forbid")


def _ensure_not_blank(value: str, field_name: str) -> str:
    """校验字符串不是空白内容，并返回去掉首尾空白后的值。"""

    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} 不能为空")
    return cleaned


class AskRequest(ApiModel):
    """POST /ask 的请求体。

    top_k 允许请求级覆盖，便于调试或临时调整检索数量；为 None 时使用系统配置。
    retrieved_chunks 和 trace 默认不返回，避免普通用户看到过多内部细节。
    """

    question: str = Field(description="用户问题")
    top_k: int | None = Field(
        default=None, ge=1, le=50, description="本次请求覆盖的检索数量"
    )
    include_retrieved_chunks: bool = Field(
        default=False, description="是否返回原始检索片段"
    )
    debug_trace: bool = Field(default=False, description="是否返回 pipeline trace")

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """问题不能为空白字符串。"""

        return _ensure_not_blank(value, "question")


class SearchRequest(ApiModel):
    """POST /search 的请求体。

    /search 只做检索，不做上下文组织和回答生成，适合排查召回质量。
    """

    query: str = Field(description="检索查询")
    top_k: int | None = Field(
        default=None, ge=1, le=50, description="本次请求覆盖的检索数量"
    )
    retriever: RetrievalStrategy | None = Field(
        default=None, description="本次请求指定的检索策略"
    )
    debug_trace: bool = Field(default=False, description="是否返回检索 trace")

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """查询不能为空白字符串。"""

        return _ensure_not_blank(value, "query")

    @field_validator("retriever")
    @classmethod
    def validate_retriever(cls, value: str | None) -> str | None:
        """检索策略名称可扩展，但不能为空白字符串。"""

        return None if value is None else _ensure_not_blank(value, "retriever")


class CompareSearchRequest(ApiModel):
    """POST /search/compare 的请求体。

    retrievers 默认比较三个内置策略；后续注册外部策略后，也可以传入外部策略名。
    """

    query: str = Field(description="检索查询")
    retrievers: list[RetrievalStrategy] = Field(
        default_factory=lambda: ["vector", "bm25", "hybrid"],
        min_length=1,
        max_length=10,
        description="本次需要并列比较的检索策略",
    )
    top_k: int | None = Field(
        default=None, ge=1, le=50, description="每个策略返回的检索数量"
    )
    debug_trace: bool = Field(default=False, description="是否返回 compare trace 和子 trace")

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """查询不能为空白字符串。"""

        return _ensure_not_blank(value, "query")

    @field_validator("retrievers")
    @classmethod
    def validate_retrievers(cls, value: list[str]) -> list[str]:
        """策略列表不能为空、不能包含空白项，也不允许重复。"""

        cleaned_retrievers: list[str] = []
        seen: set[str] = set()
        for retriever in value:
            cleaned = _ensure_not_blank(retriever, "retriever")
            if cleaned in seen:
                raise ValueError(f"retrievers 中存在重复策略：{cleaned}")
            seen.add(cleaned)
            cleaned_retrievers.append(cleaned)
        return cleaned_retrievers


class DocumentIngestRequest(ApiModel):
    """POST /documents/ingest 的请求体。"""

    source_dir: str = Field(description="待导入文档目录")
    rebuild: bool = Field(default=False, description="是否强制重建索引")

    @field_validator("source_dir")
    @classmethod
    def validate_source_dir(cls, value: str) -> str:
        """文档目录不能为空白字符串。"""

        return _ensure_not_blank(value, "source_dir")


class CitationResponse(ApiModel):
    """回答引用来源。"""

    citation_id: str
    chunk_id: str
    doc_id: str
    version_id: str
    title: str | None = None
    source_path: str
    snippet: str
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None


class RetrievalSignalResponse(ApiModel):
    """单个召回源对检索结果提供的证据。"""

    retriever: str
    rank: int
    score: float


class RetrievedChunkResponse(ApiModel):
    """API 返回的检索片段。"""

    chunk_id: str
    doc_id: str
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
    metadata: dict[str, Any] = Field(default_factory=dict)
    retrieval_signals: list[RetrievalSignalResponse] = Field(default_factory=list)


class TraceStageResponse(ApiModel):
    """单个 pipeline 阶段的 trace 信息。"""

    stage: str
    status: Literal["success", "error"]
    latency_ms: float
    detail: dict[str, Any] = Field(default_factory=dict)


class TraceResponse(ApiModel):
    """一次请求的完整 trace 信息。"""

    trace_id: str
    final_status: Literal["running", "success", "error"]
    latency_ms: float
    failure_type: str | None = None
    error_message: str | None = None
    stages: list[TraceStageResponse] = Field(default_factory=list)


class AskResponse(ApiModel):
    """POST /ask 的响应体。"""

    answer: str
    citations: list[CitationResponse] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunkResponse] = Field(default_factory=list)
    trace_id: str
    latency_ms: float
    trace: TraceResponse | None = None


class SearchResponse(ApiModel):
    """POST /search 的响应体。"""

    query: str
    results: list[RetrievedChunkResponse] = Field(default_factory=list)
    trace_id: str
    top_k: int
    retriever: RetrievalStrategy
    latency_ms: float
    trace: TraceResponse | None = None


class ComparedChunkOverlapResponse(ApiModel):
    """多个策略共同命中的 chunk。"""

    chunk_id: str
    retrievers: list[RetrievalStrategy]
    ranks_by_retriever: dict[RetrievalStrategy, int]


class ComparedStrategyResponse(ApiModel):
    """某个策略在 compare search 中的执行结果。"""

    retriever: RetrievalStrategy
    status: Literal["success", "error"]
    results: list[RetrievedChunkResponse] = Field(default_factory=list)
    trace_id: str | None = None
    latency_ms: float | None = None
    report_path: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    trace: TraceResponse | None = None


class CompareSearchResponse(ApiModel):
    """POST /search/compare 的响应体。"""

    query: str
    top_k: int
    retrievers: list[RetrievalStrategy]
    status: Literal["success", "partial_error", "error"]
    strategy_results: list[ComparedStrategyResponse] = Field(default_factory=list)
    overlaps: list[ComparedChunkOverlapResponse] = Field(default_factory=list)
    trace_id: str
    latency_ms: float
    trace: TraceResponse | None = None


class DocumentSummaryResponse(ApiModel):
    """文档列表中的单个文档摘要。"""

    doc_id: str
    version_id: str
    title: str | None = None
    source_path: str
    content_hash: str
    chunk_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentListResponse(ApiModel):
    """GET /documents 的响应体。"""

    documents: list[DocumentSummaryResponse] = Field(default_factory=list)
    total: int


class DocumentIngestResponse(ApiModel):
    """POST /documents/ingest 的响应体。"""

    index_id: str
    document_count: int
    chunk_count: int
    vector_count: int
    manifest: dict[str, Any]
    trace_id: str


class HealthResponse(ApiModel):
    """GET /health 的响应体。"""

    status: Literal["ok"] = "ok"
    service: str = "paper-rag-assistant"


class ErrorResponse(ApiModel):
    """统一错误响应体。

    trace_id 放在响应体里，方便命令行、日志和前端直接读取。
    真正接入 HTTP 框架后，也可以同时复制到 X-Trace-Id 响应头。
    """

    code: str
    message: str
    trace_id: str | None = None
    detail: dict[str, Any] | None = None


def citation_to_response(citation: Citation) -> CitationResponse:
    """把领域层 Citation 转换成 API 响应模型。"""

    return CitationResponse(
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


def retrieved_chunk_to_response(chunk: RetrievedChunk) -> RetrievedChunkResponse:
    """把领域层 RetrievedChunk 转换成 API 响应模型。"""

    return RetrievedChunkResponse(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        version_id=chunk.version_id,
        text=chunk.text,
        score=chunk.score,
        rank=chunk.rank,
        retriever=chunk.retriever,
        source_path=chunk.source_path,
        chunk_index=chunk.chunk_index,
        title=chunk.title,
        section=chunk.section,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        metadata=chunk.metadata,
        retrieval_signals=[
            RetrievalSignalResponse(
                retriever=signal.retriever,
                rank=signal.rank,
                score=signal.score,
            )
            for signal in chunk.retrieval_signals
        ],
    )


def trace_to_response(trace: RagTrace) -> TraceResponse:
    """把领域层 RagTrace 转换成 API 响应模型。"""

    return TraceResponse(
        trace_id=trace.trace_id,
        final_status=trace.final_status,
        latency_ms=trace.latency_ms,
        failure_type=trace.failure_type,
        error_message=trace.error_message,
        stages=[
            TraceStageResponse(
                stage=stage.stage,
                status=stage.status,
                latency_ms=stage.latency_ms,
                detail=stage.detail,
            )
            for stage in trace.stages
        ],
    )


def rag_answer_to_response(
    answer: RagAnswer,
    *,
    include_retrieved_chunks: bool = False,
    trace: RagTrace | None = None,
) -> AskResponse:
    """把领域层 RagAnswer 转换成 /ask 响应。

    retrieved_chunks 默认不暴露；当用户打开调试或显式要求时再返回。
    """

    return AskResponse(
        answer=answer.answer,
        citations=[citation_to_response(citation) for citation in answer.citations],
        retrieved_chunks=[
            retrieved_chunk_to_response(chunk) for chunk in answer.retrieved_chunks
        ]
        if include_retrieved_chunks
        else [],
        trace_id=answer.trace_id,
        latency_ms=answer.latency_ms,
        trace=trace_to_response(trace) if trace is not None else None,
    )


def compare_search_result_to_response(
    result: RetrievalComparisonResult,
    *,
    debug_trace: bool = False,
) -> CompareSearchResponse:
    """把 retrieval 层 compare search 结果转换成 API 响应。"""

    return CompareSearchResponse(
        query=result.query,
        top_k=result.top_k,
        retrievers=list(result.retrievers),
        status=result.status,
        strategy_results=[
            ComparedStrategyResponse(
                retriever=strategy_result.retriever,
                status=strategy_result.status,
                results=[
                    retrieved_chunk_to_response(chunk)
                    for chunk in strategy_result.results
                ],
                trace_id=(
                    strategy_result.trace.trace_id
                    if strategy_result.trace is not None
                    else None
                ),
                latency_ms=(
                    strategy_result.trace.latency_ms
                    if strategy_result.trace is not None
                    else None
                ),
                report_path=(
                    strategy_result.report_path.as_posix()
                    if strategy_result.report_path is not None
                    else None
                ),
                error_code=strategy_result.error_code,
                error_message=strategy_result.error_message,
                trace=(
                    trace_to_response(strategy_result.trace)
                    if debug_trace and strategy_result.trace is not None
                    else None
                ),
            )
            for strategy_result in result.strategy_results
        ],
        overlaps=[
            ComparedChunkOverlapResponse(
                chunk_id=overlap.chunk_id,
                retrievers=list(overlap.retrievers),
                ranks_by_retriever=overlap.ranks_by_retriever,
            )
            for overlap in result.overlaps
        ],
        trace_id=result.trace.trace_id,
        latency_ms=result.trace.latency_ms,
        trace=trace_to_response(result.trace) if debug_trace else None,
    )


__all__ = [
    "AskRequest",
    "AskResponse",
    "CompareSearchRequest",
    "CompareSearchResponse",
    "ComparedChunkOverlapResponse",
    "ComparedStrategyResponse",
    "CitationResponse",
    "DocumentIngestRequest",
    "DocumentIngestResponse",
    "DocumentListResponse",
    "DocumentSummaryResponse",
    "ErrorResponse",
    "HealthResponse",
    "RetrievalSignalResponse",
    "RetrievedChunkResponse",
    "SearchRequest",
    "SearchResponse",
    "TraceResponse",
    "TraceStageResponse",
    "compare_search_result_to_response",
    "citation_to_response",
    "rag_answer_to_response",
    "retrieved_chunk_to_response",
    "trace_to_response",
]
