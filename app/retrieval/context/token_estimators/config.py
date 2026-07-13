"""Token estimator 运行时配置。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenEstimatorConfig:
    """选择上下文 token 估算策略。"""

    strategy: str = "regex"

    def __post_init__(self) -> None:
        normalized_strategy = self.strategy.strip()
        if not normalized_strategy:
            raise ValueError("token estimator strategy 不能为空")
        object.__setattr__(self, "strategy", normalized_strategy)
