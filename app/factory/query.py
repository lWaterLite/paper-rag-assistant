"""查询规划对象组装。"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.errors import AppError, ErrorCode
from app.factory.configs import ConfigFactory
from app.factory.llm import LlmFactory
from app.retrieval.query import (
    QueryPlannerRegistry,
    QueryPlanningStage,
    build_default_query_planner_registry,
)


@dataclass(slots=True)
class QueryFactory:
    """根据统一配置与 LLM 依赖构造查询规划阶段。"""

    configs: ConfigFactory
    llm: LlmFactory
    registry: QueryPlannerRegistry = field(
        default_factory=build_default_query_planner_registry
    )

    def build_query_planning_stage(self) -> QueryPlanningStage:
        """创建由 Registry 解析策略的查询规划阶段。"""

        config = self.configs.generation.query_planning
        try:
            planner = self.registry.create(
                config,
                llm_client=self.llm.client,
                model=self.configs.generation.llm.model,
                timeout_seconds=self.configs.generation.llm.timeout_seconds,
            )
        except ValueError as exc:
            raise AppError(ErrorCode.INVALID_CONFIG, str(exc)) from exc
        return QueryPlanningStage(config=config, planner=planner)
