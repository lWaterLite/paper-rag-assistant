"""Rerank 检索后处理阶段。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from app.core.errors import AppError, ErrorCode
from app.retrieval.models import RerankSignal, RetrievedChunk
from app.retrieval.pipeline_types import RetrievalPipelineContext, RetrievalStageResult
from app.retrieval.rerankers.base import Reranker
from app.retrieval.rerankers.config import RerankingConfig


class RerankStage:
    """调用已注入的 reranker 并将结果写回运行时检索对象。"""

    def __init__(self, *, reranker: Reranker, config: RerankingConfig) -> None:
        if not config.enabled:
            raise ValueError("禁用 rerank 时不应构造 RerankStage")
        self._reranker = reranker
        self._config = config

    def process(
        self,
        chunks: Sequence[RetrievedChunk],
        context: RetrievalPipelineContext,
    ) -> RetrievalStageResult:
        """执行重排序；fail-open 时保留原始排序并报告降级。"""

        try:
            reranked = self._reranker.rerank(
                context.query,
                chunks,
                limit=len(chunks),
            )
            ranked_chunks = [
                replace(
                    item.chunk,
                    rank=rank,
                    rerank_signal=RerankSignal(
                        reranker=self._reranker.name,
                        rank=rank,
                        score=item.score,
                    ),
                )
                for rank, item in enumerate(reranked, start=1)
            ]
            self._validate_reranked_candidates(chunks, ranked_chunks)
        except (AppError, OSError, RuntimeError, TypeError, ValueError) as exc:
            if self._config.failure_mode == "fail_open":
                return RetrievalStageResult(
                    chunks=list(chunks),
                    detail={
                        "reranker": self._reranker.name,
                        "failure_mode": self._config.failure_mode,
                        "degraded": True,
                        "error_message": str(exc),
                    },
                )
            raise AppError(
                ErrorCode.RERANK_FAILED,
                f"reranker {self._reranker.name} 执行失败：{exc}",
            ) from exc

        return RetrievalStageResult(
            chunks=ranked_chunks,
            detail={
                "reranker": self._reranker.name,
                "failure_mode": self._config.failure_mode,
                "degraded": False,
            },
        )

    @staticmethod
    def _validate_reranked_candidates(
        original_chunks: Sequence[RetrievedChunk],
        reranked_chunks: Sequence[RetrievedChunk],
    ) -> None:
        """拒绝外部 reranker 返回未知或重复候选。"""

        original_chunk_ids = {chunk.chunk_id for chunk in original_chunks}
        reranked_chunk_ids = [chunk.chunk_id for chunk in reranked_chunks]
        unknown_chunk_ids = set(reranked_chunk_ids) - original_chunk_ids
        if unknown_chunk_ids:
            raise AppError(
                ErrorCode.RERANK_FAILED,
                "reranker 返回了候选集合外的 chunk："
                + ", ".join(sorted(unknown_chunk_ids)),
            )
        if len(reranked_chunk_ids) != len(set(reranked_chunk_ids)):
            raise AppError(ErrorCode.RERANK_FAILED, "reranker 返回了重复 chunk")
        if set(reranked_chunk_ids) != original_chunk_ids:
            missing_chunk_ids = original_chunk_ids - set(reranked_chunk_ids)
            raise AppError(
                ErrorCode.RERANK_FAILED,
                "reranker 没有返回完整候选集：" + ", ".join(sorted(missing_chunk_ids)),
            )
