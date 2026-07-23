"""RAG pipeline 编排。

这里把在线问答流程串起来：检索 -> 上下文组织 -> 生成。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, NoReturn, Protocol

from app.core.errors import AppError, ErrorCode
from app.core.tracing import RagTrace
from app.generation.answering import AnswerGenerator
from app.generation.models import RagAnswer
from app.retrieval.context import ContextPackRequest, ContextPacker
from app.retrieval.context.evidence_transformers.models import EvidenceTransformRequest
from app.retrieval.context.evidence_transformers.stage import EvidenceTransformStage
from app.retrieval.pipeline import RetrievalPipelineResult
from app.retrieval.query import QueryPlanningStage


class RetrievalService(Protocol):
    """RAG pipeline 需要的 retrieval 应用服务接口。"""

    def search_queries(
        self,
        query: str,
        *,
        retrieval_queries: tuple[str, ...],
        top_k: int | None = None,
        retriever: str | None = None,
    ) -> RetrievalPipelineResult:
        """执行统一 retrieval pipeline。"""


@dataclass(frozen=True, slots=True)
class RagPipelineConfig:
    """在线 RAG pipeline 的运行时配置。"""

    top_k: int = 3

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("RAG pipeline top_k 必须大于 0")


class RagPipeline:
    """在线 RAG 问答 pipeline。"""

    def __init__(
        self,
        config: RagPipelineConfig,
        *,
        retrieval_service: RetrievalService,
        context_packer: ContextPacker,
        evidence_transform_stage: EvidenceTransformStage,
        query_planning_stage: QueryPlanningStage,
        answer_generator: AnswerGenerator,
    ) -> None:
        self._config = config
        self._retrieval_service = retrieval_service
        self._context_packer = context_packer
        self._evidence_transform_stage = evidence_transform_stage
        self._query_planning_stage = query_planning_stage
        self._answer_generator = answer_generator

    def ask(self, question: str, *, top_k: int | None = None) -> RagAnswer:
        """根据用户问题执行一次 RAG 问答。"""

        trace = RagTrace()

        started = time.perf_counter()
        try:
            query_plan = self._query_planning_stage.plan(question)
        except Exception as exc:
            self._record_failure_and_raise(
                trace=trace,
                stage="query_planning",
                started_at=started,
                exc=exc,
                default_code=ErrorCode.QUERY_REWRITE_FAILED,
                detail={"question": question},
            )
        else:
            trace.record_stage(
                "query_planning",
                "success",
                started,
                query_plan.to_trace_detail(),
            )

        started = time.perf_counter()
        try:
            retrieval_result = self._retrieval_service.search_queries(
                query_plan.original_query,
                retrieval_queries=query_plan.retrieval_queries,
                top_k=self._config.top_k if top_k is None else top_k,
            )
            retrieved_chunks = retrieval_result.results
        except Exception as exc:
            self._record_failure_and_raise(
                trace=trace,
                stage="retrieval",
                started_at=started,
                exc=exc,
                default_code=ErrorCode.RETRIEVAL_FAILED,
                detail={
                    "query": query_plan.original_query,
                    "top_k": self._config.top_k if top_k is None else top_k,
                    "retrieval_trace_id": exc.trace_id
                    if isinstance(exc, AppError)
                    else None,
                },
            )
        else:
            trace.record_stage(
                "retrieval",
                "success",
                started,
                {
                    "query": query_plan.original_query,
                    "retrieval_queries": list(query_plan.retrieval_queries),
                    "top_k": self._config.top_k if top_k is None else top_k,
                    "returned": len(retrieved_chunks),
                    "retrieval_trace_id": retrieval_result.trace.trace_id,
                    "retrieval_report_path": (
                        retrieval_result.report_path.as_posix()
                        if retrieval_result.report_path is not None
                        else None
                    ),
                },
            )

        started = time.perf_counter()
        try:
            evidence_result = self._evidence_transform_stage.process(
                EvidenceTransformRequest(query=question, chunks=retrieved_chunks)
            )
            evidence_candidates = evidence_result.candidates
        except Exception as exc:
            self._record_failure_and_raise(
                trace=trace,
                stage="evidence_transformation",
                started_at=started,
                exc=exc,
                default_code=ErrorCode.EVIDENCE_TRANSFORM_FAILED,
                detail={"retrieved_chunks": len(retrieved_chunks)},
            )
        else:
            trace.record_stage(
                "evidence_transformation",
                "success",
                started,
                {
                    **evidence_result.detail,
                    "input_count": len(retrieved_chunks),
                    "output_count": len(evidence_candidates),
                },
            )

        started = time.perf_counter()
        try:
            packed_context = self._context_packer.pack(
                ContextPackRequest(query=question, candidates=evidence_candidates)
            )
        except Exception as exc:
            self._record_failure_and_raise(
                trace=trace,
                stage="context_packing",
                started_at=started,
                exc=exc,
                default_code=ErrorCode.RETRIEVAL_FAILED,
                detail={
                    "retrieved_chunks": len(retrieved_chunks),
                    "evidence_candidates": len(evidence_candidates),
                },
            )
        else:
            trace.record_stage(
                "context_packing",
                "success",
                started,
                {
                    "used_chunks": len(packed_context.used_chunks),
                    "dropped_chunks": len(packed_context.dropped_chunks),
                    "citation_count": len(packed_context.citations),
                    "context_chars": len(packed_context.context_text),
                    "context_tokens": packed_context.token_usage.used_context_tokens,
                    "available_context_tokens": (
                        packed_context.token_usage.available_context_tokens
                    ),
                    "dropped_chunk_count": len(packed_context.dropped_chunks),
                },
            )

        started = time.perf_counter()
        try:
            answer = self._answer_generator.generate(
                question=question,
                packed_context=packed_context,
                retrieved_chunks=retrieved_chunks,
                trace=trace,
            )
        except Exception as exc:
            self._record_failure_and_raise(
                trace=trace,
                stage="generation",
                started_at=started,
                exc=exc,
                default_code=ErrorCode.GENERATION_FAILED,
                detail={
                    "retrieved_chunks": len(retrieved_chunks),
                    "citations": len(packed_context.citations),
                },
            )
        else:
            generation_detail: dict[str, object] = {
                "answer_chars": len(answer.answer),
                "status": answer.status,
            }
            if answer.diagnostics is not None:
                generation_detail.update(
                    {
                        "provider": answer.diagnostics.provider,
                        "model": answer.diagnostics.model,
                        "prompt_tokens": answer.diagnostics.prompt_tokens,
                        "output_tokens": answer.diagnostics.output_tokens,
                        "citation_validation": answer.diagnostics.citation_validation,
                    }
                )
            trace.record_stage(
                "generation", "success", started, generation_detail
            )
            trace.mark_success()

        return replace(answer, trace=trace, latency_ms=trace.latency_ms)

    def _record_failure_and_raise(
        self,
        *,
        trace: RagTrace,
        stage: str,
        started_at: float,
        exc: Exception,
        default_code: ErrorCode,
        detail: dict[str, Any] | None = None,
    ) -> NoReturn:
        """记录阶段失败，并抛出带 trace_id 的 AppError。"""

        error_code = exc.code if isinstance(exc, AppError) else default_code
        error_message = exc.message if isinstance(exc, AppError) else str(exc)
        stage_detail = {
            **(detail or {}),
            "error_code": error_code.value,
            "error_message": error_message,
        }

        trace.record_stage(stage, "error", started_at, stage_detail)
        trace.mark_failed(failure_type=stage, error_message=error_message)

        raise AppError(
            code=error_code,
            message=f"{stage} 阶段失败：{error_message}",
            trace_id=trace.trace_id,
            trace=trace,
        ) from exc
