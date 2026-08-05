"""不改写查询的基线规划器。"""

from __future__ import annotations

from app.retrieval.query.models import QueryPlan


class PassthroughQueryPlanner:
    """原样使用用户问题，作为禁用或降级时的安全基线。"""

    def plan(self, question: str) -> QueryPlan:
        """返回只包含原始 query 的计划。"""

        cleaned_question = question.strip()
        return QueryPlan(
            original_query=cleaned_question,
            primary_query=cleaned_question,
            strategy="passthrough",
        )
