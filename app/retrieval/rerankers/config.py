"""重排序运行时配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


RerankFailureMode = Literal["fail_open", "fail_closed"]


@dataclass(frozen=True, slots=True)
class RerankingConfig:
    """Rerank 阶段真正接收的运行时配置。"""

    enabled: bool = False
    strategy: str = "lexical"
    candidate_limit: int = 12
    batch_size: int = 8
    failure_mode: RerankFailureMode = "fail_open"

    def __post_init__(self) -> None:
        normalized_strategy = self.strategy.strip()
        if not normalized_strategy:
            raise ValueError("reranking strategy 不能为空")
        if self.candidate_limit <= 0:
            raise ValueError("reranking candidate_limit 必须大于 0")
        if self.batch_size <= 0:
            raise ValueError("reranking batch_size 必须大于 0")
        object.__setattr__(self, "strategy", normalized_strategy)
