"""查询规划阶段的领域模型。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QueryPlan:
    """从原始问题派生出的可追溯检索计划。"""

    original_query: str
    primary_query: str
    additional_queries: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    hyde_document: str | None = None
    strategy: str = "passthrough"
    fallback_used: bool = False
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        original_query = self.original_query.strip()
        primary_query = self.primary_query.strip()
        if not original_query:
            raise ValueError("original_query 不能为空")
        if not primary_query:
            raise ValueError("primary_query 不能为空")

        additional_queries = _normalize_distinct_values(
            self.additional_queries,
            excluded={primary_query},
        )
        keywords = _normalize_distinct_values(self.keywords)
        hyde_document = self.hyde_document.strip() if self.hyde_document else None
        strategy = self.strategy.strip()
        if not strategy:
            raise ValueError("query plan strategy 不能为空")

        object.__setattr__(self, "original_query", original_query)
        object.__setattr__(self, "primary_query", primary_query)
        object.__setattr__(self, "additional_queries", additional_queries)
        object.__setattr__(self, "keywords", keywords)
        object.__setattr__(self, "hyde_document", hyde_document or None)
        object.__setattr__(self, "strategy", strategy)

    @property
    def retrieval_queries(self) -> tuple[str, ...]:
        """返回用于常规检索的 query，且保持稳定顺序。"""

        return (self.primary_query, *self.additional_queries)

    def to_trace_detail(self) -> dict[str, object]:
        """返回不包含资料正文的安全 trace 摘要。"""

        return {
            "strategy": self.strategy,
            "primary_query": self.primary_query,
            "additional_query_count": len(self.additional_queries),
            "keyword_count": len(self.keywords),
            "hyde_enabled": self.hyde_document is not None,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
        }


def _normalize_distinct_values(
    values: tuple[str, ...],
    *,
    excluded: set[str] | None = None,
) -> tuple[str, ...]:
    """清理空白值和重复项，并保持输入顺序。"""

    seen = set(excluded or set())
    normalized: list[str] = []
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            normalized.append(cleaned)
    return tuple(normalized)
