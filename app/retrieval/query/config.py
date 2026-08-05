"""查询规划的运行期配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class QueryPlanningConfig:
    """控制查询改写、多查询与 HyDE 的运行期策略。"""

    enabled: bool = True
    strategy: str = "rule_based"
    multi_query_enabled: bool = False
    max_additional_queries: int = 2
    hyde_enabled: bool = False
    failure_mode: Literal["fail_open", "fail_closed"] = "fail_open"

    def __post_init__(self) -> None:
        strategy = self.strategy.strip()
        if not strategy:
            raise ValueError("query planning strategy 不能为空")
        if self.max_additional_queries < 0:
            raise ValueError("max_additional_queries 不能小于 0")
        object.__setattr__(self, "strategy", strategy)
