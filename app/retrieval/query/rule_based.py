"""面向论文术语的可解释查询规划器。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.retrieval.query.config import QueryPlanningConfig
from app.retrieval.query.models import QueryPlan


_TERM_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("重排序", ("reranking", "reranker")),
    ("精排", ("reranking", "reranker")),
    ("两阶段", ("two-stage retrieval", "candidate retrieval")),
    ("混合检索", ("hybrid retrieval", "dense sparse retrieval")),
    ("向量检索", ("dense retrieval", "embedding similarity")),
    ("关键词检索", ("BM25", "sparse retrieval")),
    ("引用", ("citation", "grounded answer")),
    ("幻觉", ("hallucination", "groundedness")),
    ("上下文", ("context packing", "token budget")),
)


@dataclass(frozen=True, slots=True)
class RuleBasedQueryPlanner:
    """通过受控术语映射补充中英文论文关键词。

    它不尝试推断新事实，适合作为没有外部模型时的可解释默认策略。
    """

    config: QueryPlanningConfig

    def plan(self, question: str) -> QueryPlan:
        """生成主 query、可选补充 query 与关键词。"""

        cleaned_question = _normalize_whitespace(question)
        keywords = _match_keywords(cleaned_question)
        primary_query = " ".join((cleaned_question, *keywords)).strip()
        additional_queries = (
            tuple(keywords[: self.config.max_additional_queries])
            if self.config.multi_query_enabled
            else ()
        )
        return QueryPlan(
            original_query=cleaned_question,
            primary_query=primary_query,
            additional_queries=additional_queries,
            keywords=tuple(keywords),
            strategy="rule_based",
        )


def _normalize_whitespace(value: str) -> str:
    """把用户输入压缩为稳定的单空格表达。"""

    normalized = re.sub(r"\s+", " ", value).strip()
    if not normalized:
        raise ValueError("用户问题不能为空")
    return normalized


def _match_keywords(question: str) -> list[str]:
    """从受控词表中选择与问题明确相关的英文术语。"""

    lowered_question = question.lower()
    matched: list[str] = []
    for trigger, aliases in _TERM_ALIASES:
        if trigger in question:
            for alias in aliases:
                if alias.lower() not in lowered_question and alias not in matched:
                    matched.append(alias)
    return matched
