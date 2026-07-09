"""Retrieval 子系统内部 pipeline。"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from app.core.errors import AppError, ErrorCode
from app.core.models import RagTrace, RetrievedChunk
from app.retrieval.configs import RetrievalConfig, RetrievalStrategy
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
        result_stages: Sequence[RetrievalResultStage] | None = None,
    ) -> None:
        self._registry = registry
        self._config = config
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

        cleaned_query = query.strip()
        if not cleaned_query:
            raise AppError(ErrorCode.RETRIEVAL_FAILED, "检索 query 不能为空")

        resolved_top_k = self._resolve_top_k(top_k)
        resolved_retriever = retriever or self._config.strategy
        try:
            retriever_impl = self._registry.resolve(resolved_retriever)
        except ValueError as exc:
            raise AppError(
                ErrorCode.INVALID_CONFIG,
                str(exc),
            ) from exc

        context = RetrievalPipelineContext(
            query=cleaned_query,
            retriever=resolved_retriever,
            top_k=resolved_top_k,
        )
        trace = RagTrace()
        started = time.perf_counter()

        try:
            results = retriever_impl.retrieve(cleaned_query, top_k=resolved_top_k)
            results = self._apply_result_stages(results, context)
        except Exception as exc:
            error_code = (
                exc.code if isinstance(exc, AppError) else ErrorCode.RETRIEVAL_FAILED
            )
            error_message = exc.message if isinstance(exc, AppError) else str(exc)
            trace.record_stage(
                "retrieval",
                "error",
                started,
                {
                    "query": cleaned_query,
                    "top_k": resolved_top_k,
                    "retriever": resolved_retriever,
                    "error_code": error_code.value,
                    "error_message": error_message,
                },
            )
            trace.mark_failed("retrieval", error_message)
            raise AppError(
                error_code,
                f"search 检索失败：{error_message}",
                trace_id=trace.trace_id,
                trace=trace,
            ) from exc

        trace.record_stage(
            "retrieval",
            "success",
            started,
            {
                "query": cleaned_query,
                "top_k": resolved_top_k,
                "retriever": resolved_retriever,
                "returned": len(results),
            },
        )
        trace.mark_success()

        return RetrievalPipelineResult(
            query=cleaned_query,
            retriever=resolved_retriever,
            top_k=resolved_top_k,
            results=results,
            trace=trace,
        )

    def _resolve_top_k(self, top_k: int | None) -> int:
        """解析本次请求使用的 top_k。"""

        if top_k is None:
            resolved = self._config.top_k
        else:
            resolved = top_k
        if resolved <= 0:
            raise AppError(
                ErrorCode.INVALID_CONFIG, f"top_k 必须大于 0，当前 top_k={resolved}"
            )
        return resolved

    def _apply_result_stages(
        self,
        results: Sequence[RetrievedChunk],
        context: RetrievalPipelineContext,
    ) -> list[RetrievedChunk]:
        """按顺序执行检索结果后处理阶段。"""

        processed = list(results)
        for stage in self._result_stages:
            processed = stage.process(processed, context)
        return processed

    @staticmethod
    def _build_default_stages(config: RetrievalConfig) -> list[RetrievalResultStage]:
        """根据配置创建默认后处理阶段。"""

        stages: list[RetrievalResultStage] = []
        if config.deduplicate_by_chunk_id:
            stages.append(ChunkIdDeduplicationStage())
        stages.append(TopKLimitStage())
        return stages
