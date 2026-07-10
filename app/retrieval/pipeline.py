"""Retrieval 子系统内部 pipeline。"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from app.core.errors import AppError, ErrorCode
from app.core.models import RagTrace, RetrievedChunk
from app.retrieval.configs import RetrievalConfig, RetrievalStrategy
from app.retrieval.reporting import (
    RetrievalExecutionReport,
    RetrievalReportWriteResult,
    RetrievalReporter,
    RetrievalStageObservation,
)
from app.retrieval.retrievers.registry import RetrieverRegistry


@dataclass(frozen=True)
class RetrievalPipelineContext:
    """一次 retrieval pipeline 运行时上下文。"""

    query: str
    retriever: RetrievalStrategy
    top_k: int


@dataclass(frozen=True)
class RetrievalPipelineResult:
    """一次 retrieval pipeline 的领域层结果。"""

    query: str
    retriever: RetrievalStrategy
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
    ) -> list[RetrievedChunk]:
        """处理当前阶段接收到的检索结果。"""


class ChunkIdDeduplicationStage:
    """按 chunk_id 去重，并保持首次出现顺序。"""

    def process(
        self,
        chunks: Sequence[RetrievedChunk],
        context: RetrievalPipelineContext,
    ) -> list[RetrievedChunk]:
        """去除重复 chunk，并重新分配 rank。"""

        _ = context
        seen: set[str] = set()
        unique_chunks: list[RetrievedChunk] = []
        for chunk in chunks:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            unique_chunks.append(chunk)

        return [
            replace(chunk, rank=rank)
            for rank, chunk in enumerate(unique_chunks, start=1)
        ]


class TopKLimitStage:
    """限制最终返回的 top_k 数量。"""

    def process(
        self,
        chunks: Sequence[RetrievedChunk],
        context: RetrievalPipelineContext,
    ) -> list[RetrievedChunk]:
        """只保留本次请求需要的前 top_k 条结果。"""

        return list(chunks[: context.top_k])


class RetrievalPipeline:
    """封装一次完整检索流程。

    这里负责 query 校验、检索器选择、检索执行、结果后处理和 trace 记录。
    API 层或应用服务层不直接拼这些步骤，避免 retrieval 逻辑散落在多个入口中。
    """

    def __init__(
        self,
        *,
        registry: RetrieverRegistry,
        config: RetrievalConfig,
        reporter: RetrievalReporter,
        result_stages: Sequence[RetrievalResultStage] | None = None,
    ) -> None:
        self._registry = registry
        self._config = config
        if config.top_k is None:
            raise ValueError("retrieval config top_k 不能为空")
        self._default_top_k: int = config.top_k
        self._reporter = reporter
        self._result_stages = (
            list(result_stages)
            if result_stages is not None
            else self._build_default_stages(config)
        )

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        retriever: RetrievalStrategy | None = None,
    ) -> RetrievalPipelineResult:
        """执行一次完整检索。"""

        trace = RagTrace()
        cleaned_query = query.strip()
        resolved_top_k: int | None = None
        resolved_retriever: str | None = None
        candidate_count = 0
        deduplicated_count = 0
        latest_results: list[RetrievedChunk] = []
        observations: list[RetrievalStageObservation] = []
        retrieval_started = time.perf_counter()

        try:
            if not cleaned_query:
                raise AppError(ErrorCode.RETRIEVAL_FAILED, "检索 query 不能为空")

            active_top_k = self._resolve_top_k(top_k)
            resolved_top_k = active_top_k
            active_retriever = (
                self._config.strategy if retriever is None else retriever.strip()
            )
            resolved_retriever = active_retriever
            try:
                retriever_impl = self._registry.resolve(active_retriever)
            except ValueError as exc:
                raise AppError(ErrorCode.INVALID_CONFIG, str(exc)) from exc

            context = RetrievalPipelineContext(
                query=cleaned_query,
                retriever=active_retriever,
                top_k=active_top_k,
            )

            started = time.perf_counter()
            latest_results = retriever_impl.retrieve(
                cleaned_query,
                top_k=active_top_k,
            )
            candidate_count = len(latest_results)
            deduplicated_count = candidate_count
            observations.append(
                RetrievalStageObservation(
                    stage="retriever_execution",
                    input_count=0,
                    output_count=candidate_count,
                    latency_ms=_elapsed_ms(started),
                )
            )
            (
                latest_results,
                stage_observations,
                deduplicated_count,
            ) = self._apply_result_stages(latest_results, context)
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
            self._record_report_write(
                trace,
                report_write_result,
                reporting_started,
            )
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
                "retriever": resolved_retriever,
                "candidate_count": candidate_count,
                "deduplicated_count": deduplicated_count,
                "returned_count": len(latest_results),
            },
        )
        trace.mark_success()

        if resolved_top_k is None or resolved_retriever is None:
            raise RuntimeError("retrieval 内部错误：成功执行后缺少解析配置")

        reporting_started = time.perf_counter()
        report_write_result = self._write_execution_report(
            query=cleaned_query,
            requested_top_k=top_k,
            resolved_top_k=resolved_top_k,
            requested_retriever=retriever,
            resolved_retriever=resolved_retriever,
            candidate_count=candidate_count,
            deduplicated_count=deduplicated_count,
            results=latest_results,
            observations=observations,
            trace=trace,
        )
        self._record_report_write(
            trace,
            report_write_result,
            reporting_started,
        )
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
            top_k=resolved_top_k,
            results=latest_results,
            trace=trace,
            report_path=report_write_result.path,
        )

    def _resolve_top_k(self, top_k: int | None) -> int:
        """解析本次请求使用的 top_k。"""

        resolved: int
        if top_k is None:
            resolved = self._default_top_k
        else:
            resolved = top_k
        if resolved <= 0:
            raise AppError(
                ErrorCode.INVALID_CONFIG, f"top_k 必须大于 0，当前 top_k={resolved}"
            )
        return resolved

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

    def _apply_result_stages(
        self,
        results: Sequence[RetrievedChunk],
        context: RetrievalPipelineContext,
    ) -> tuple[
        list[RetrievedChunk],
        list[RetrievalStageObservation],
        int,
    ]:
        """按顺序执行检索结果后处理阶段。"""

        processed = list(results)
        observations: list[RetrievalStageObservation] = []
        deduplicated_count = len(processed)
        for stage in self._result_stages:
            input_count = len(processed)
            started = time.perf_counter()
            processed = stage.process(processed, context)
            observations.append(
                RetrievalStageObservation(
                    stage=type(stage).__name__,
                    input_count=input_count,
                    output_count=len(processed),
                    latency_ms=_elapsed_ms(started),
                )
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

    @staticmethod
    def _build_default_stages(config: RetrievalConfig) -> list[RetrievalResultStage]:
        """根据配置创建默认后处理阶段。"""

        stages: list[RetrievalResultStage] = []
        if config.deduplicate_by_chunk_id:
            stages.append(ChunkIdDeduplicationStage())
        stages.append(TopKLimitStage())
        return stages


def _elapsed_ms(started: float) -> float:
    """计算阶段耗时并统一报告精度。"""

    return round((time.perf_counter() - started) * 1000, 2)
