"""查询规划阶段及其失败降级策略。"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.errors import AppError, ErrorCode
from app.retrieval.query.base import QueryPlanner
from app.retrieval.query.config import QueryPlanningConfig
from app.retrieval.query.models import QueryPlan
from app.retrieval.query.passthrough import PassthroughQueryPlanner


@dataclass(frozen=True, slots=True)
class QueryPlanningStage:
    """执行查询规划，并把失败语义限制在检索入口。"""

    config: QueryPlanningConfig
    planner: QueryPlanner

    def plan(self, question: str) -> QueryPlan:
        """返回查询计划；允许时回退到原始 query。"""

        cleaned_question = question.strip()
        if not cleaned_question:
            raise AppError(ErrorCode.QUERY_REWRITE_FAILED, "用户问题不能为空")
        if not self.config.enabled:
            return PassthroughQueryPlanner().plan(cleaned_question)

        try:
            return self.planner.plan(cleaned_question)
        except Exception as exc:
            if self.config.failure_mode == "fail_open":
                fallback = PassthroughQueryPlanner().plan(cleaned_question)
                return QueryPlan(
                    original_query=fallback.original_query,
                    primary_query=fallback.primary_query,
                    strategy=self.config.strategy,
                    fallback_used=True,
                    fallback_reason=str(exc),
                )
            raise AppError(
                ErrorCode.QUERY_REWRITE_FAILED,
                f"查询改写失败：{exc}",
            ) from exc
