"""检索与问答 API 契约。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator

from app.api.contracts.common import ApiModel, TraceResponse, ensure_not_blank
from app.retrieval.configuration import RetrievalStrategy


class AskRequest(ApiModel):
    """POST /ask 的请求体。"""

    question: str = Field(description="用户问题")
    top_k: int | None = Field(default=None, ge=1, le=50, description="检索数量")
    include_retrieved_chunks: bool = Field(default=False, description="是否返回检索片段")
    debug_trace: bool = Field(default=False, description="是否返回流程追踪")

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """问题不能为空白字符串。"""

        return ensure_not_blank(value, "question")


class SearchRequest(ApiModel):
    """POST /search 的请求体。"""

    query: str = Field(description="检索查询")
    top_k: int | None = Field(default=None, ge=1, le=50, description="检索数量")
    retriever: RetrievalStrategy | None = Field(default=None, description="检索策略")
    debug_trace: bool = Field(default=False, description="是否返回检索追踪")

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """查询不能为空白字符串。"""

        return ensure_not_blank(value, "query")

    @field_validator("retriever")
    @classmethod
    def validate_retriever(cls, value: str | None) -> str | None:
        """策略名称可扩展，但不能为空白字符串。"""

        return None if value is None else ensure_not_blank(value, "retriever")


class CompareSearchRequest(ApiModel):
    """POST /search/compare 的请求体。"""

    query: str = Field(description="检索查询")
    retrievers: list[RetrievalStrategy] = Field(
        default_factory=lambda: ["vector", "bm25", "hybrid"],
        min_length=1,
        max_length=10,
        description="需要比较的检索策略",
    )
    top_k: int | None = Field(default=None, ge=1, le=50, description="每个策略的检索数量")
    debug_trace: bool = Field(default=False, description="是否返回比较追踪")

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """查询不能为空白字符串。"""

        return ensure_not_blank(value, "query")

    @field_validator("retrievers")
    @classmethod
    def validate_retrievers(cls, value: list[str]) -> list[str]:
        """策略列表不允许空白项或重复项。"""

        cleaned_retrievers: list[str] = []
        seen: set[str] = set()
        for retriever in value:
            cleaned = ensure_not_blank(retriever, "retriever")
            if cleaned in seen:
                raise ValueError(f"retrievers 中存在重复策略：{cleaned}")
            seen.add(cleaned)
            cleaned_retrievers.append(cleaned)
        return cleaned_retrievers


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


class RerankSignalResponse(ApiModel):
    """重排序阶段提供的运行时证据。"""

    reranker: str
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
    rerank_signal: RerankSignalResponse | None = None


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
    """单个策略在比较检索中的执行结果。"""

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
    report_path: str | None = None
    trace: TraceResponse | None = None
