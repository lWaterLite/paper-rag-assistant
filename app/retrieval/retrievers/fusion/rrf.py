"""Reciprocal Rank Fusion 实现。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.retrieval.models import RetrievalSignal, RetrievedChunk
from app.retrieval.retrievers.fusion.base import (
    FusedRetrievalHit,
    RankedResultSet,
)


@dataclass(slots=True)
class _RrfAccumulator:
    """单个 chunk 的 RRF 聚合状态。"""

    chunk: RetrievedChunk
    first_seen_order: int
    score: float = 0.0
    signals: tuple[RetrievalSignal, ...] = ()


class ReciprocalRankFusion:
    """使用各召回源的排名进行加权倒数排名融合。"""

    def __init__(self, rank_constant: int = 60) -> None:
        if rank_constant <= 0:
            raise ValueError("RRF rank_constant 必须大于 0")
        self._rank_constant = rank_constant

    def fuse(
        self,
        result_sets: Sequence[RankedResultSet],
        *,
        limit: int,
    ) -> list[FusedRetrievalHit]:
        """按 chunk_id 聚合多路排名并计算 RRF 分数。"""

        if limit <= 0:
            return []

        accumulators: dict[str, _RrfAccumulator] = {}
        next_seen_order = 0

        for result_set in result_sets:
            seen_in_source: set[str] = set()
            for rank, chunk in enumerate(result_set.chunks, start=1):
                if chunk.chunk_id in seen_in_source:
                    continue
                seen_in_source.add(chunk.chunk_id)

                accumulator = accumulators.get(chunk.chunk_id)
                if accumulator is None:
                    accumulator = _RrfAccumulator(
                        chunk=chunk,
                        first_seen_order=next_seen_order,
                    )
                    accumulators[chunk.chunk_id] = accumulator
                    next_seen_order += 1

                accumulator.score += result_set.weight / (
                    self._rank_constant + rank
                )
                accumulator.signals += (
                    RetrievalSignal(
                        retriever=result_set.source,
                        rank=rank,
                        score=chunk.score,
                    ),
                )

        ranked = sorted(
            accumulators.values(),
            key=lambda item: (-item.score, item.first_seen_order, item.chunk.chunk_id),
        )
        return [
            FusedRetrievalHit(
                chunk=item.chunk,
                score=item.score,
                signals=item.signals,
            )
            for item in ranked[:limit]
        ]
