"""检索应用服务。"""

from __future__ import annotations

from collections.abc import Sequence

from app.retrieval.configs import RetrievalConfig, RetrievalStrategy
from app.retrieval.comparison import RetrievalComparisonResult
from app.retrieval.pipeline import (
    RetrievalComparisonPipeline,
    RetrievalPipeline,
    RetrievalPipelineResult,
)
from app.retrieval.reporting import RetrievalComparisonReporter, RetrievalReporter
from app.retrieval.rerankers import Reranker, RerankingConfig
from app.retrieval.retrievers.registry import RetrieverRegistry


SearchResult = RetrievalPipelineResult
CompareSearchResult = RetrievalComparisonResult


class SearchService:
    """只执行检索、不生成回答的在线应用服务。

    具体检索流程由 RetrievalPipeline 负责，这里保留为 API handler 面向的应用服务入口。
    """

    def __init__(
        self,
        *,
        registry: RetrieverRegistry,
        config: RetrievalConfig,
        reranking_config: RerankingConfig,
        reranker: Reranker | None,
        reporter: RetrievalReporter,
    ) -> None:
        self._pipeline = RetrievalPipeline(
            registry=registry,
            config=config,
            reranking_config=reranking_config,
            reranker=reranker,
            reporter=reporter,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        retriever: RetrievalStrategy | None = None,
    ) -> SearchResult:
        """执行一次检索并返回可调试结果。"""

        return self._pipeline.search(query, top_k=top_k, retriever=retriever)


class CompareSearchService:
    """执行多策略检索比较的 retrieval 子系统服务。"""

    def __init__(
        self,
        *,
        registry: RetrieverRegistry,
        config: RetrievalConfig,
        reranking_config: RerankingConfig,
        reranker: Reranker | None,
        reporter: RetrievalReporter,
        comparison_reporter: RetrievalComparisonReporter,
    ) -> None:
        search_pipeline = RetrievalPipeline(
            registry=registry,
            config=config,
            reranking_config=reranking_config,
            reranker=reranker,
            reporter=reporter,
        )
        self._pipeline = RetrievalComparisonPipeline(
            search_pipeline=search_pipeline,
            config=config,
            reporter=comparison_reporter,
        )

    def compare(
        self,
        query: str,
        *,
        retrievers: Sequence[RetrievalStrategy],
        top_k: int | None = None,
    ) -> CompareSearchResult:
        """执行多策略检索比较。"""

        return self._pipeline.compare(
            query,
            retrievers=retrievers,
            top_k=top_k,
        )
