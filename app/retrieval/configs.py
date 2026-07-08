"""检索子系统运行时配置。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


RetrievalStrategy = Literal["vector", "bm25", "hybrid"]


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
