"""检索领域结果到 API 响应契约的转换。"""

from __future__ import annotations

from app.api.contracts.common import TraceResponse, TraceStageResponse
from app.api.contracts.retrieval import (
    AskResponse,
    CitationResponse,
    ComparedChunkOverlapResponse,
    ComparedStrategyResponse,
    CompareSearchResponse,
    RerankSignalResponse,
    RetrievedChunkResponse,
    RetrievalSignalResponse,
)
from app.core.models import Citation, RagAnswer, RetrievedChunk
from app.core.tracing import RagTrace
from app.retrieval.comparison import RetrievalComparisonResult


def citation_to_response(citation: Citation) -> CitationResponse:
    """把领域 Citation 转换成 API 响应。"""

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
    """把领域检索结果转换成 API 响应。"""

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
                retriever=signal.retriever, rank=signal.rank, score=signal.score
            )
            for signal in chunk.retrieval_signals
        ],
        rerank_signal=(
            RerankSignalResponse(
                reranker=chunk.rerank_signal.reranker,
                rank=chunk.rerank_signal.rank,
                score=chunk.rerank_signal.score,
            )
            if chunk.rerank_signal is not None
            else None
        ),
    )


def trace_to_response(trace: RagTrace) -> TraceResponse:
    """把领域追踪转换成 API 响应。"""

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
    """把问答结果转换成 /ask 响应。"""

    return AskResponse(
        answer=answer.answer,
        citations=[citation_to_response(citation) for citation in answer.citations],
        retrieved_chunks=(
            [retrieved_chunk_to_response(chunk) for chunk in answer.retrieved_chunks]
            if include_retrieved_chunks
            else []
        ),
        trace_id=answer.trace_id,
        latency_ms=answer.latency_ms,
        trace=trace_to_response(trace) if trace is not None else None,
    )


def compare_search_result_to_response(
    result: RetrievalComparisonResult,
    *,
    debug_trace: bool = False,
) -> CompareSearchResponse:
    """把 comparison 结果转换成 API 响应。"""

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
        report_path=result.report_path.as_posix() if result.report_path is not None else None,
        trace=trace_to_response(result.trace) if debug_trace else None,
    )
