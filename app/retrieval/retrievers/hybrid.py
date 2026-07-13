"""组合多个召回源的 Hybrid Retriever。"""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.core.models import RetrievedChunk
from app.retrieval.configuration import HybridRetrievalConfig
from app.retrieval.retrievers.base import Retriever
from app.retrieval.retrievers.fusion import (
    FusionStrategy,
    RankedResultSet,
)


@dataclass(frozen=True, slots=True)
class HybridRetrievalSource:
    """Hybrid Retriever 使用的单个召回源。"""

    name: str
    retriever: Retriever
    weight: float

    def __post_init__(self) -> None:
        normalized_name = self.name.strip()
        if not normalized_name:
            raise ValueError("hybrid 召回源名称不能为空")
        if self.weight <= 0:
            raise ValueError("hybrid 召回源权重必须大于 0")
        object.__setattr__(self, "name", normalized_name)


class HybridRetriever:
    """扩大各路候选集，通过融合策略生成最终检索结果。"""

    def __init__(
        self,
        *,
        sources: tuple[HybridRetrievalSource, ...],
        fusion_strategy: FusionStrategy,
        config: HybridRetrievalConfig,
    ) -> None:
        if len(sources) < 2:
            raise ValueError("HybridRetriever 至少需要两个召回源")
        source_names = [source.name for source in sources]
        if len(source_names) != len(set(source_names)):
            raise ValueError("HybridRetriever 召回源名称不能重复")

        self._sources = sources
        self._fusion_strategy = fusion_strategy
        self._config = config

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """分别召回候选结果，完成融合并返回最终 top-k。"""

        if top_k <= 0:
            return []

        candidate_k = top_k * self._config.candidate_multiplier
        result_sets = [
            RankedResultSet(
                source=source.name,
                weight=source.weight,
                chunks=source.retriever.retrieve(query, top_k=candidate_k),
            )
            for source in self._sources
        ]
        fused_hits = self._fusion_strategy.fuse(result_sets, limit=top_k)

        return [
            replace(
                hit.chunk,
                score=round(hit.score, 6),
                rank=rank,
                retriever="hybrid",
                retrieval_signals=hit.signals,
            )
            for rank, hit in enumerate(fused_hits, start=1)
        ]
