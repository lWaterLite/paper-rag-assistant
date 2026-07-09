"""检索结果融合策略的基础结构。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.core.models import RetrievalSignal, RetrievedChunk


@dataclass(frozen=True, slots=True)
class RankedResultSet:
    """一个召回源返回的有序候选集合。"""

    source: str
    weight: float
    chunks: Sequence[RetrievedChunk]

    def __post_init__(self) -> None:
        normalized_source = self.source.strip()
        if not normalized_source:
            raise ValueError("融合结果源名称不能为空")
        if self.weight <= 0:
            raise ValueError("融合结果源权重必须大于 0")
        object.__setattr__(self, "source", normalized_source)


@dataclass(frozen=True, slots=True)
class FusedRetrievalHit:
    """融合策略产生的内部命中结果。"""

    chunk: RetrievedChunk
    score: float
    signals: tuple[RetrievalSignal, ...]


class FusionStrategy(Protocol):
    """多路检索结果融合策略协议。"""

    def fuse(
        self,
        result_sets: Sequence[RankedResultSet],
        *,
        limit: int,
    ) -> list[FusedRetrievalHit]:
        """融合多个有序候选集合，并返回最多 limit 条结果。"""
