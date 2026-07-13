"""检索子系统运行时配置。"""

from __future__ import annotations

from dataclasses import dataclass, field

RetrievalStrategy = str


@dataclass(frozen=True)
class BM25Config:
    """BM25 检索器配置。"""

    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        if self.k1 <= 0:
            raise ValueError("bm25 k1 必须大于 0")
        if not 0 <= self.b <= 1:
            raise ValueError("bm25 b 必须在 0 到 1 之间")


@dataclass(frozen=True, slots=True)
class HybridRetrievalConfig:
    """Hybrid Retriever 的运行时配置。"""

    candidate_multiplier: int = 3
    rrf_rank_constant: int = 60
    vector_weight: float = 1.0
    bm25_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.candidate_multiplier < 1:
            raise ValueError("hybrid candidate_multiplier 必须大于等于 1")
        if self.rrf_rank_constant <= 0:
            raise ValueError("hybrid rrf_rank_constant 必须大于 0")
        if self.vector_weight <= 0:
            raise ValueError("hybrid vector_weight 必须大于 0")
        if self.bm25_weight <= 0:
            raise ValueError("hybrid bm25_weight 必须大于 0")


@dataclass(frozen=True)
class RetrievalConfig:
    """检索服务运行时配置。"""

    strategy: RetrievalStrategy = "vector"
    top_k: int = 3
    bm25: BM25Config = field(default_factory=BM25Config)
    deduplicate_by_chunk_id: bool = True

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("retrieval top_k 必须大于 0")
