"""检索应用服务。"""

from __future__ import annotations

from app.retrieval.configs import RetrievalConfig, RetrievalStrategy
from app.retrieval.pipeline import RetrievalPipeline, RetrievalPipelineResult
from app.retrieval.reporting import RetrievalReporter
from app.retrieval.retrievers.registry import RetrieverRegistry


SearchResult = RetrievalPipelineResult


class SearchService:
    """只执行检索、不生成回答的在线应用服务。

    具体检索流程由 RetrievalPipeline 负责，这里保留为 API handler 面向的应用服务入口。
    """

    def __init__(
        self,
        *,
        registry: RetrieverRegistry,
        config: RetrievalConfig,
        reporter: RetrievalReporter,
    ) -> None:
        self._pipeline = RetrievalPipeline(
            registry=registry,
            config=config,
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
