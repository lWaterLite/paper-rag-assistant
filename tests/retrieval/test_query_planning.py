"""查询规划策略与失败降级测试。"""

from __future__ import annotations

import json
import unittest

from app.core.errors import AppError, ErrorCode
from app.llm import LlmRequest, LlmResponse
from app.retrieval.query.config import QueryPlanningConfig
from app.retrieval.query.llm import LlmQueryPlanner
from app.retrieval.query.rule_based import RuleBasedQueryPlanner
from app.retrieval.query.stage import QueryPlanningStage


class JsonLlmClient:
    """返回固定 JSON 的测试 LLM Client。"""

    @property
    def provider_name(self) -> str:
        return "test"

    def complete(self, request: LlmRequest) -> LlmResponse:
        _ = request
        return LlmResponse(
            content=json.dumps(
                {
                    "primary_query": "cross-encoder reranking latency",
                    "additional_queries": ["two-stage retrieval cost"],
                    "keywords": ["cross-encoder", "reranking"],
                    "hyde_document": "A cross-encoder reranks a small candidate set.",
                }
            ),
            model="test",
        )


class FailingPlanner:
    """模拟不可用的查询规划器。"""

    def plan(self, question: str):
        _ = question
        raise RuntimeError("模型超时")


class QueryPlanningTest(unittest.TestCase):
    """验证 QueryPlan 保留原始问题和明确的失败语义。"""

    def test_rule_based_planner_adds_controlled_paper_terms(self) -> None:
        planner = RuleBasedQueryPlanner(
            QueryPlanningConfig(
                multi_query_enabled=True,
                max_additional_queries=2,
            )
        )

        plan = planner.plan("两阶段检索中的重排序为什么更慢？")

        self.assertEqual(plan.original_query, "两阶段检索中的重排序为什么更慢？")
        self.assertIn("reranking", plan.primary_query)
        self.assertLessEqual(len(plan.additional_queries), 2)
        self.assertFalse(plan.fallback_used)

    def test_llm_planner_returns_bounded_multi_query_and_hyde_fields(self) -> None:
        planner = LlmQueryPlanner(
            config=QueryPlanningConfig(
                strategy="llm",
                multi_query_enabled=True,
                max_additional_queries=1,
                hyde_enabled=True,
            ),
            llm_client=JsonLlmClient(),
            model="test",
            timeout_seconds=10,
        )

        plan = planner.plan("为什么需要精排？")

        self.assertEqual(plan.primary_query, "cross-encoder reranking latency")
        self.assertEqual(plan.additional_queries, ("two-stage retrieval cost",))
        self.assertIsNotNone(plan.hyde_document)

    def test_stage_falls_back_to_original_query_when_configured_fail_open(self) -> None:
        stage = QueryPlanningStage(
            config=QueryPlanningConfig(failure_mode="fail_open"),
            planner=FailingPlanner(),
        )

        plan = stage.plan("  原始问题  ")

        self.assertTrue(plan.fallback_used)
        self.assertEqual(plan.primary_query, "原始问题")
        self.assertIn("模型超时", plan.fallback_reason)

    def test_stage_raises_domain_error_when_configured_fail_closed(self) -> None:
        stage = QueryPlanningStage(
            config=QueryPlanningConfig(failure_mode="fail_closed"),
            planner=FailingPlanner(),
        )

        with self.assertRaises(AppError) as context:
            stage.plan("原始问题")

        self.assertEqual(context.exception.code, ErrorCode.QUERY_REWRITE_FAILED)


if __name__ == "__main__":
    unittest.main()
