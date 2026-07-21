"""Retrieval 子系统内部 pipeline。"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from app.core.errors import AppError, ErrorCode
from app.retrieval.models import RetrievedChunk
from app.core.tracing import RagTrace
from app.retrieval.comparison import (
    ComparedChunkOverlap,
    ComparedStrategyResult,
    RetrievalComparisonResult,
)
from app.retrieval.configuration import RetrievalConfig, RetrievalStrategy
from app.retrieval.pipeline_types import RetrievalPipelineContext, RetrievalStageResult
from app.retrieval.reporting import (
    RetrievalComparisonExecutionReport,
    RetrievalComparisonOverlapReport,
    RetrievalComparisonReporter,
    RetrievalComparisonStrategyReport,
    RetrievalExecutionReport,
    RetrievalReportWriteResult,
    RetrievalReporter,
    RetrievalStageObservation,
)
from app.retrieval.rerankers import Reranker, RerankingConfig, RerankStage
from app.retrieval.retrievers.registry import RetrieverRegistry


@dataclass(frozen=True, slots=True)
class RetrievalPipelineResult:
    """一次 retrieval pipeline 的领域层结果。"""

    query: str
    retriever: RetrievalStrategy
    candidate_limit: int
    top_k: int
    results: list[RetrievedChunk]
    trace: RagTrace
    report_path: Path | None = None


class RetrievalResultStage(Protocol):
    """检索结果后处理阶段协议。"""

    def process(
        self,
        chunks: Sequence[RetrievedChunk],
        context: RetrievalPipelineContext,
    ) -> RetrievalStageResult:
        """处理当前阶段接收到的检索结果。"""


class ChunkIdDeduplicationStage:
    """按 chunk_id 去重，并保持首次出现顺序。"""

    def process(
        self,
        chunks: Sequence[RetrievedChunk],
        context: RetrievalPipelineContext,
    ) -> RetrievalStageResult:
        """去除重复 chunk，并重新分配 rank。"""

        _ = context
        seen: set[str] = set()
        unique_chunks: list[RetrievedChunk] = []
        for chunk in chunks:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            unique_chunks.append(chunk)

        return RetrievalStageResult(
            chunks=[
                replace(chunk, rank=rank)
                for rank, chunk in enumerate(unique_chunks, start=1)
            ],
            detail={"duplicates_removed": len(chunks) - len(unique_chunks)},
        )


class TopKLimitStage:
    """限制最终返回的 top_k 数量。"""

    def process(
        self,
        chunks: Sequence[RetrievedChunk],
        context: RetrievalPipelineContext,
    ) -> RetrievalStageResult:
        """只保留本次请求需要的前 top_k 条结果。"""

        return RetrievalStageResult(
            chunks=list(chunks[: context.top_k]),
            detail={"final_top_k": context.top_k},
        )


class RetrievalPipeline:
    """封装候选召回、后处理、重排序与执行报告。"""

    def __init__(
        self,
        *,
        registry: RetrieverRegistry,
        config: RetrievalConfig,
        reranking_config: RerankingConfig,
        reranker: Reranker | None,
        reporter: RetrievalReporter,
        result_stages: Sequence[RetrievalResultStage] | None = None,
    ) -> None:
        if reranking_config.enabled and reranker is None:
            raise ValueError("启用 rerank 时必须显式注入 reranker")
        if not reranking_config.enabled and reranker is not None:
            raise ValueError("禁用 rerank 时不应注入 reranker")

        self._registry = registry
        self._config = config
        self._reranking_config = reranking_config
        self._reranker = reranker
        self._default_top_k = config.top_k
        self._reporter = reporter
        self._result_stages = (
            list(result_stages)
            if result_stages is not None
            else self._build_default_stages(config, reranking_config, reranker)
        )

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        retriever: RetrievalStrategy | None = None,
    ) -> RetrievalPipelineResult:
        """执行一次完整检索，并将 rerank 前后阶段记录到 trace 与报告。"""

        trace = RagTrace()
        cleaned_query = query.strip()
        resolved_top_k: int | None = None
        resolved_candidate_limit: int | None = None
        resolved_retriever: str | None = None
        candidate_count = 0
        deduplicated_count = 0
        latest_results: list[RetrievedChunk] = []
        observations: list[RetrievalStageObservation] = []
        retrieval_started = time.perf_counter()

        try:
            if not cleaned_query:
                raise AppError(ErrorCode.RETRIEVAL_FAILED, "检索 query 不能为空")

            resolved_top_k = self._resolve_top_k(top_k)
            resolved_candidate_limit = self._resolve_candidate_limit(resolved_top_k)
            resolved_retriever = (
                self._config.strategy if retriever is None else retriever.strip()
            )
            if not resolved_retriever:
                raise AppError(ErrorCode.INVALID_CONFIG, "retriever 不能为空")
            try:
                retriever_impl = self._registry.resolve(resolved_retriever)
            except ValueError as exc:
                raise AppError(ErrorCode.INVALID_CONFIG, str(exc)) from exc

            context = RetrievalPipelineContext(
                query=cleaned_query,
                retriever=resolved_retriever,
                candidate_limit=resolved_candidate_limit,
                top_k=resolved_top_k,
            )
            retriever_started = time.perf_counter()
            latest_results = retriever_impl.retrieve(
                cleaned_query,
                top_k=resolved_candidate_limit,
            )
            candidate_count = len(latest_results)
            deduplicated_count = candidate_count
            retriever_detail = {
                "retriever": resolved_retriever,
                "candidate_limit": resolved_candidate_limit,
            }
            observations.append(
                RetrievalStageObservation(
                    stage="retriever_execution",
                    status="success",
                    input_count=0,
                    output_count=candidate_count,
                    latency_ms=_elapsed_ms(retriever_started),
                    detail=retriever_detail,
                )
            )
            trace.record_stage(
                "retriever_execution",
                "success",
                retriever_started,
                {**retriever_detail, "returned_count": candidate_count},
            )
            latest_results, stage_observations, deduplicated_count = self._apply_result_stages(
                latest_results,
                context,
                trace,
            )
            observations.extend(stage_observations)
        except Exception as exc:
            error_code = (
                exc.code if isinstance(exc, AppError) else ErrorCode.RETRIEVAL_FAILED
            )
            error_message = exc.message if isinstance(exc, AppError) else str(exc)
            trace.record_stage(
                "retrieval",
                "error",
                retrieval_started,
                {
                    "query": cleaned_query,
                    "requested_top_k": top_k,
                    "resolved_top_k": resolved_top_k,
                    "resolved_candidate_limit": resolved_candidate_limit,
                    "requested_retriever": retriever,
                    "resolved_retriever": resolved_retriever,
                    "candidate_count": candidate_count,
                    "returned_count": len(latest_results),
                    "error_code": error_code.value,
                    "error_message": error_message,
                },
            )
            trace.mark_failed("retrieval", error_message)
            reporting_started = time.perf_counter()
            report_write_result = self._write_execution_report(
                query=cleaned_query,
                requested_top_k=top_k,
                resolved_top_k=resolved_top_k,
                resolved_candidate_limit=resolved_candidate_limit,
                requested_retriever=retriever,
                resolved_retriever=resolved_retriever,
                candidate_count=candidate_count,
                deduplicated_count=deduplicated_count,
                results=latest_results,
                observations=observations,
                trace=trace,
                error_code=error_code.value,
                error_message=error_message,
            )
            self._record_report_write(trace, report_write_result, reporting_started)
            raise AppError(
                error_code,
                f"search 检索失败：{error_message}",
                trace_id=trace.trace_id,
                trace=trace,
            ) from exc

        trace.record_stage(
            "retrieval",
            "success",
            retrieval_started,
            {
                "query": cleaned_query,
                "top_k": resolved_top_k,
                "candidate_limit": resolved_candidate_limit,
                "retriever": resolved_retriever,
                "candidate_count": candidate_count,
                "deduplicated_count": deduplicated_count,
                "returned_count": len(latest_results),
            },
        )
        trace.mark_success()

        if (
            resolved_top_k is None
            or resolved_candidate_limit is None
            or resolved_retriever is None
        ):
            raise RuntimeError("retrieval 内部错误：成功执行后缺少解析配置")

        reporting_started = time.perf_counter()
        report_write_result = self._write_execution_report(
            query=cleaned_query,
            requested_top_k=top_k,
            resolved_top_k=resolved_top_k,
            resolved_candidate_limit=resolved_candidate_limit,
            requested_retriever=retriever,
            resolved_retriever=resolved_retriever,
            candidate_count=candidate_count,
            deduplicated_count=deduplicated_count,
            results=latest_results,
            observations=observations,
            trace=trace,
        )
        self._record_report_write(trace, report_write_result, reporting_started)
        if report_write_result.error_message and report_write_result.fatal:
            trace.mark_failed("retrieval_reporting", report_write_result.error_message)
            raise AppError(
                ErrorCode.RETRIEVAL_FAILED,
                f"retrieval 报告写入失败：{report_write_result.error_message}",
                trace_id=trace.trace_id,
                trace=trace,
            )

        return RetrievalPipelineResult(
            query=cleaned_query,
            retriever=resolved_retriever,
            candidate_limit=resolved_candidate_limit,
            top_k=resolved_top_k,
            results=latest_results,
            trace=trace,
            report_path=report_write_result.path,
        )

    def _resolve_top_k(self, top_k: int | None) -> int:
        """解析本次请求最终返回的 top_k。"""

        resolved = self._default_top_k if top_k is None else top_k
        if resolved <= 0:
            raise AppError(
                ErrorCode.INVALID_CONFIG,
                f"top_k 必须大于 0，当前 top_k={resolved}",
            )
        return resolved

    def _resolve_candidate_limit(self, top_k: int) -> int:
        """解析候选召回上限，关闭 rerank 时保持原有 top-k 行为。"""

        if not self._reranking_config.enabled:
            return top_k
        return max(top_k, self._reranking_config.candidate_limit)

    def _apply_result_stages(
        self,
        results: Sequence[RetrievedChunk],
        context: RetrievalPipelineContext,
        trace: RagTrace,
    ) -> tuple[list[RetrievedChunk], list[RetrievalStageObservation], int]:
        """按顺序执行后处理阶段，并记录每个阶段的事实。"""

        processed = list(results)
        observations: list[RetrievalStageObservation] = []
        deduplicated_count = len(processed)
        for stage in self._result_stages:
            stage_name = type(stage).__name__
            input_count = len(processed)
            started = time.perf_counter()
            try:
                stage_result = stage.process(processed, context)
            except Exception as exc:
                error_message = exc.message if isinstance(exc, AppError) else str(exc)
                detail = {"error_message": error_message}
                observations.append(
                    RetrievalStageObservation(
                        stage=stage_name,
                        status="error",
                        input_count=input_count,
                        output_count=input_count,
                        latency_ms=_elapsed_ms(started),
                        detail=detail,
                    )
                )
                trace.record_stage(stage_name, "error", started, detail)
                raise

            processed = stage_result.chunks
            detail = dict(stage_result.detail)
            observations.append(
                RetrievalStageObservation(
                    stage=stage_name,
                    status="success",
                    input_count=input_count,
                    output_count=len(processed),
                    latency_ms=_elapsed_ms(started),
                    detail=detail,
                )
            )
            trace.record_stage(
                stage_name,
                "success",
                started,
                {**detail, "input_count": input_count, "output_count": len(processed)},
            )
            if isinstance(stage, ChunkIdDeduplicationStage):
                deduplicated_count = len(processed)
        return processed, observations, deduplicated_count

    def _write_execution_report(
        self,
        *,
        query: str,
        requested_top_k: int | None,
        resolved_top_k: int | None,
        resolved_candidate_limit: int | None,
        requested_retriever: str | None,
        resolved_retriever: str | None,
        candidate_count: int,
        deduplicated_count: int,
        results: Sequence[RetrievedChunk],
        observations: Sequence[RetrievalStageObservation],
        trace: RagTrace,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> RetrievalReportWriteResult:
        """把当前执行状态交给 reporter，pipeline 不处理 JSON 细节。"""

        return self._reporter.write(
            RetrievalExecutionReport(
                query=query,
                requested_top_k=requested_top_k,
                resolved_top_k=resolved_top_k,
                resolved_candidate_limit=resolved_candidate_limit,
                requested_retriever=requested_retriever,
                resolved_retriever=resolved_retriever,
                candidate_count=candidate_count,
                deduplicated_count=deduplicated_count,
                returned_count=len(results),
                stage_observations=tuple(observations),
                results=tuple(results),
                runtime=self._reporter.runtime_snapshot,
                trace=trace,
                error_code=error_code,
                error_message=error_message,
            )
        )

    def _record_report_write(
        self,
        trace: RagTrace,
        result: RetrievalReportWriteResult,
        started: float,
    ) -> None:
        """把报告写入结果记录到调用方可见的 trace。"""

        if not self._reporter.enabled:
            return
        if result.error_message is not None:
            trace.record_stage(
                "retrieval_reporting",
                "error",
                started,
                {"error_message": result.error_message, "fatal": result.fatal},
            )
            return
        trace.record_stage(
            "retrieval_reporting",
            "success",
            started,
            {
                "report_path": result.path.as_posix()
                if result.path is not None
                else None
            },
        )

    @staticmethod
    def _build_default_stages(
        config: RetrievalConfig,
        reranking_config: RerankingConfig,
        reranker: Reranker | None,
    ) -> list[RetrievalResultStage]:
        """构建默认后处理顺序：去重、重排、最终截断。"""

        stages: list[RetrievalResultStage] = []
        if config.deduplicate_by_chunk_id:
            stages.append(ChunkIdDeduplicationStage())
        if reranking_config.enabled:
            if reranker is None:
                raise ValueError("启用 rerank 时必须显式注入 reranker")
            stages.append(RerankStage(reranker=reranker, config=reranking_config))
        stages.append(TopKLimitStage())
        return stages


class RetrievalComparisonPipeline:
    """编排多策略检索比较流程。"""

    def __init__(
        self,
        *,
        search_pipeline: RetrievalPipeline,
        config: RetrievalConfig,
        reporter: RetrievalComparisonReporter,
    ) -> None:
        self._search_pipeline = search_pipeline
        self._default_top_k = config.top_k
        self._reporter = reporter

    def compare(
        self,
        query: str,
        *,
        retrievers: Sequence[RetrievalStrategy],
        top_k: int | None = None,
    ) -> RetrievalComparisonResult:
        """并列执行多个单策略检索，并保留每个子请求的独立结果。"""

        trace = RagTrace()
        started = time.perf_counter()
        cleaned_query = query.strip()
        if not cleaned_query:
            raise AppError(ErrorCode.RETRIEVAL_FAILED, "检索 query 不能为空")

        active_top_k = self._resolve_top_k(top_k)
        active_retrievers = self._normalize_retrievers(retrievers)
        strategy_results: list[ComparedStrategyResult] = []

        for strategy in active_retrievers:
            strategy_started = time.perf_counter()
            try:
                result = self._search_pipeline.search(
                    cleaned_query,
                    top_k=active_top_k,
                    retriever=strategy,
                )
            except AppError as exc:
                child_trace = exc.trace if isinstance(exc.trace, RagTrace) else None
                strategy_results.append(
                    ComparedStrategyResult(
                        retriever=strategy,
                        status="error",
                        trace=child_trace,
                        error_code=exc.code.value,
                        error_message=exc.message,
                    )
                )
                trace.record_stage(
                    "compare_strategy",
                    "error",
                    strategy_started,
                    {
                        "retriever": strategy,
                        "error_code": exc.code.value,
                        "error_message": exc.message,
                        "child_trace_id": (
                            child_trace.trace_id if child_trace is not None else None
                        ),
                    },
                )
                continue

            strategy_results.append(
                ComparedStrategyResult(
                    retriever=strategy,
                    status="success",
                    results=tuple(result.results),
                    trace=result.trace,
                    report_path=result.report_path,
                )
            )
            trace.record_stage(
                "compare_strategy",
                "success",
                strategy_started,
                {
                    "retriever": strategy,
                    "candidate_limit": result.candidate_limit,
                    "returned_count": len(result.results),
                    "child_trace_id": result.trace.trace_id,
                    "report_path": (
                        result.report_path.as_posix()
                        if result.report_path is not None
                        else None
                    ),
                },
            )

        status = self._resolve_status(strategy_results)
        overlaps = self._build_overlaps(strategy_results)
        trace.record_stage(
            "retrieval_comparison",
            "success" if status != "error" else "error",
            started,
            {
                "query": cleaned_query,
                "top_k": active_top_k,
                "retrievers": list(active_retrievers),
                "status": status,
                "success_count": sum(
                    result.status == "success" for result in strategy_results
                ),
                "failure_count": sum(
                    result.status == "error" for result in strategy_results
                ),
                "overlap_count": len(overlaps),
            },
        )
        if status == "error":
            trace.mark_failed("retrieval_comparison", "所有检索策略都执行失败")
        else:
            trace.mark_success()

        comparison_result = RetrievalComparisonResult(
            query=cleaned_query,
            top_k=active_top_k,
            retrievers=tuple(active_retrievers),
            status=status,
            strategy_results=tuple(strategy_results),
            overlaps=tuple(overlaps),
            trace=trace,
        )
        reporting_started = time.perf_counter()
        report_write_result = self._write_comparison_report(comparison_result)
        self._record_comparison_report_write(
            trace,
            report_write_result,
            reporting_started,
        )
        if report_write_result.error_message and report_write_result.fatal:
            trace.mark_failed(
                "retrieval_comparison_reporting",
                report_write_result.error_message,
            )
            raise AppError(
                ErrorCode.RETRIEVAL_FAILED,
                f"compare search 报告写入失败：{report_write_result.error_message}",
                trace_id=trace.trace_id,
                trace=trace,
            )
        return replace(comparison_result, report_path=report_write_result.path)

    def _write_comparison_report(
        self,
        result: RetrievalComparisonResult,
    ) -> RetrievalReportWriteResult:
        """把 compare search 的父级状态交给聚合 reporter。"""

        return self._reporter.write(
            RetrievalComparisonExecutionReport(
                query=result.query,
                top_k=result.top_k,
                retrievers=result.retrievers,
                status=result.status,
                strategy_results=tuple(
                    RetrievalComparisonStrategyReport(
                        retriever=item.retriever,
                        status=item.status,
                        returned_count=len(item.results),
                        child_trace_id=(
                            item.trace.trace_id if item.trace is not None else None
                        ),
                        child_trace_status=(
                            item.trace.final_status if item.trace is not None else None
                        ),
                        child_latency_ms=(
                            item.trace.latency_ms if item.trace is not None else None
                        ),
                        report_path=(
                            item.report_path.as_posix()
                            if item.report_path is not None
                            else None
                        ),
                        error_code=item.error_code,
                        error_message=item.error_message,
                    )
                    for item in result.strategy_results
                ),
                overlaps=tuple(
                    RetrievalComparisonOverlapReport(
                        chunk_id=overlap.chunk_id,
                        retrievers=overlap.retrievers,
                        ranks_by_retriever=overlap.ranks_by_retriever,
                    )
                    for overlap in result.overlaps
                ),
                runtime=self._reporter.runtime_snapshot,
                trace=result.trace,
            )
        )

    def _record_comparison_report_write(
        self,
        trace: RagTrace,
        result: RetrievalReportWriteResult,
        started: float,
    ) -> None:
        """把聚合报告写入结果记录到 compare search 父 trace。"""

        if not self._reporter.enabled:
            return
        if result.error_message is not None:
            trace.record_stage(
                "retrieval_comparison_reporting",
                "error",
                started,
                {"error_message": result.error_message, "fatal": result.fatal},
            )
            return
        trace.record_stage(
            "retrieval_comparison_reporting",
            "success",
            started,
            {
                "report_path": result.path.as_posix()
                if result.path is not None
                else None
            },
        )

    def _resolve_top_k(self, top_k: int | None) -> int:
        """解析 compare search 使用的 top_k。"""

        resolved = self._default_top_k if top_k is None else top_k
        if resolved <= 0:
            raise AppError(
                ErrorCode.INVALID_CONFIG,
                f"top_k 必须大于 0，当前 top_k={resolved}",
            )
        return resolved

    @staticmethod
    def _normalize_retrievers(
        retrievers: Sequence[RetrievalStrategy],
    ) -> list[RetrievalStrategy]:
        """清洗策略名称，并拒绝空值或重复项。"""

        normalized: list[str] = []
        seen: set[str] = set()
        for retriever in retrievers:
            cleaned = retriever.strip()
            if not cleaned:
                raise AppError(ErrorCode.INVALID_CONFIG, "retriever 不能为空")
            if cleaned in seen:
                raise AppError(
                    ErrorCode.INVALID_CONFIG,
                    f"compare search 中存在重复 retriever：{cleaned}",
                )
            seen.add(cleaned)
            normalized.append(cleaned)
        if not normalized:
            raise AppError(ErrorCode.INVALID_CONFIG, "compare search 至少需要一个 retriever")
        return normalized

    @staticmethod
    def _resolve_status(
        strategy_results: Sequence[ComparedStrategyResult],
    ) -> str:
        """根据各策略执行结果计算整体比较状态。"""

        success_count = sum(result.status == "success" for result in strategy_results)
        if success_count == len(strategy_results):
            return "success"
        if success_count == 0:
            return "error"
        return "partial_error"

    @staticmethod
    def _build_overlaps(
        strategy_results: Sequence[ComparedStrategyResult],
    ) -> list[ComparedChunkOverlap]:
        """计算被多个成功策略共同命中的 chunk。"""

        ranks_by_chunk: dict[str, dict[str, int]] = {}
        for strategy_result in strategy_results:
            if strategy_result.status != "success":
                continue
            for chunk in strategy_result.results:
                strategy_ranks = ranks_by_chunk.setdefault(chunk.chunk_id, {})
                strategy_ranks.setdefault(strategy_result.retriever, chunk.rank)

        overlaps: list[ComparedChunkOverlap] = []
        for chunk_id, ranks_by_retriever in ranks_by_chunk.items():
            if len(ranks_by_retriever) < 2:
                continue
            overlaps.append(
                ComparedChunkOverlap(
                    chunk_id=chunk_id,
                    retrievers=tuple(ranks_by_retriever),
                    ranks_by_retriever=dict(ranks_by_retriever),
                )
            )
        return overlaps


def _elapsed_ms(started: float) -> float:
    """计算阶段耗时并统一报告精度。"""

    return round((time.perf_counter() - started) * 1000, 2)
