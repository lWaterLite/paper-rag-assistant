"""查询规划策略注册表。"""

from __future__ import annotations

from collections.abc import Callable

from app.llm import LlmClient
from app.retrieval.query.base import QueryPlanner
from app.retrieval.query.config import QueryPlanningConfig
from app.retrieval.query.llm import LlmQueryPlanner
from app.retrieval.query.passthrough import PassthroughQueryPlanner
from app.retrieval.query.rule_based import RuleBasedQueryPlanner


QueryPlannerBuilder = Callable[
    [QueryPlanningConfig, LlmClient, str, float], QueryPlanner
]


class QueryPlannerRegistry:
    """集中管理可配置的查询规划策略。"""

    def __init__(self) -> None:
        self._builders: dict[str, QueryPlannerBuilder] = {}

    def register(self, strategy: str, builder: QueryPlannerBuilder) -> None:
        """注册一个查询规划器构建器。"""

        normalized_strategy = _normalize_strategy(strategy)
        if normalized_strategy in self._builders:
            raise ValueError(f"QueryPlanner 策略已注册：{normalized_strategy}")
        self._builders[normalized_strategy] = builder

    def create(
        self,
        config: QueryPlanningConfig,
        *,
        llm_client: LlmClient,
        model: str,
        timeout_seconds: float,
    ) -> QueryPlanner:
        """根据 Config 创建查询规划器。"""

        try:
            builder = self._builders[config.strategy]
        except KeyError as exc:
            available = ", ".join(sorted(self._builders)) or "无"
            raise ValueError(
                f"未注册的 QueryPlanner 策略：{config.strategy}；可用策略：{available}"
            ) from exc
        return builder(config, llm_client, model, timeout_seconds)

    @property
    def strategies(self) -> tuple[str, ...]:
        """返回已注册策略。"""

        return tuple(sorted(self._builders))


def build_default_query_planner_registry() -> QueryPlannerRegistry:
    """构建内置查询规划策略注册表。"""

    registry = QueryPlannerRegistry()
    registry.register(
        "passthrough",
        lambda _config, _client, _model, _timeout: PassthroughQueryPlanner(),
    )
    registry.register(
        "rule_based",
        lambda config, _client, _model, _timeout: RuleBasedQueryPlanner(config),
    )
    registry.register(
        "llm",
        lambda config, client, model, timeout: LlmQueryPlanner(
            config=config,
            llm_client=client,
            model=model,
            timeout_seconds=timeout,
        ),
    )
    return registry


def _normalize_strategy(strategy: str) -> str:
    """清理并校验策略名称。"""

    normalized_strategy = strategy.strip()
    if not normalized_strategy:
        raise ValueError("QueryPlanner strategy 不能为空")
    return normalized_strategy
