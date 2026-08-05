"""查询规划能力协议。"""

from __future__ import annotations

from typing import Protocol

from app.retrieval.query.models import QueryPlan


class QueryPlanner(Protocol):
    """将用户问题变换为检索计划。"""

    def plan(self, question: str) -> QueryPlan:
        """保留原始意图，生成可用于检索的查询计划。"""
