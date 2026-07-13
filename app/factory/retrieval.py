"""Retrieval 相关对象组装。"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.errors import AppError, ErrorCode
from app.factory.configs import ConfigFactory
from app.indexing.index_builder import RagIndex
from app.retrieval.retrievers import (
    BM25Index,
    BM25Retriever,
    HybridRetrievalSource,
    HybridRetriever,
    Retriever,
    RetrieverRegistry,
    VectorRetriever,
)
from app.retrieval.retrievers.fusion import ReciprocalRankFusion
from app.retrieval.rerankers import (
    Reranker,
    RerankerRegistry,
    RerankingConfig,
    build_default_reranker_registry,
)
from app.retrieval.reporting import (
    RetrievalComparisonReporter,
    RetrievalComparisonReportWriter,
    RetrievalConfigSnapshot,
    RetrievalIndexSnapshot,
    RetrievalReporter,
    RetrievalReportWriter,
    RetrievalRuntimeSnapshot,
)
from app.retrieval.service import CompareSearchService, SearchService
from app.retrieval.tokenizers import (
    Tokenizer,
    TokenizerRegistry,
    build_default_tokenizer_registry,
)
from app.retrieval.token_estimators import (
    TokenEstimator,
    TokenEstimatorRegistry,
    build_default_token_estimator_registry,
)
from app.retrieval.context_packer import ContextPacker, TokenAwareContextPacker


@dataclass(slots=True)
class RetrievalFactory:
    """组装在线检索相关对象。"""

    configs: ConfigFactory
    tokenizer_registry: TokenizerRegistry = field(
        default_factory=build_default_tokenizer_registry
    )
    reranker_registry: RerankerRegistry | None = None
    token_estimator_registry: TokenEstimatorRegistry = field(
        default_factory=build_default_token_estimator_registry
    )

    @staticmethod
    def build_vector_retriever(index: RagIndex) -> VectorRetriever:
        """创建向量检索器。"""

        return VectorRetriever(
            index.embedding_client,
            index.vector_collection,
            index.chunk_collection,
        )

    def build_bm25_retriever(self, index: RagIndex) -> BM25Retriever:
        """根据当前 chunk collection 创建 BM25 检索器。"""

        retrieval_config = self.configs.build_retrieval_config()
        bm25_index = BM25Index.from_chunks(
            index.chunk_collection.iter_chunks(),
            config=retrieval_config.bm25,
            tokenizer=self.build_tokenizer(),
        )
        return BM25Retriever(bm25_index)

    def build_tokenizer(self) -> Tokenizer:
        """根据当前配置创建 BM25 使用的分词器。"""

        return self.tokenizer_registry.create(
            self.configs.build_tokenizer_config()
        )

    def build_reranker(
        self,
        config: RerankingConfig | None = None,
    ) -> Reranker | None:
        """根据配置创建 reranker；禁用时显式返回 None。"""

        active_config = (
            config if config is not None else self.configs.build_reranking_config()
        )
        if not active_config.enabled:
            return None
        active_registry = (
            self.reranker_registry
            if self.reranker_registry is not None
            else build_default_reranker_registry(self.build_tokenizer())
        )
        try:
            return active_registry.create(active_config)
        except ValueError as exc:
            raise AppError(ErrorCode.INVALID_CONFIG, str(exc)) from exc

    def build_token_estimator(self) -> TokenEstimator:
        """根据配置创建生成模型窗口使用的 token estimator。"""

        try:
            return self.token_estimator_registry.create(
                self.configs.build_token_estimator_config()
            )
        except ValueError as exc:
            raise AppError(ErrorCode.INVALID_CONFIG, str(exc)) from exc

    def build_context_packer(self) -> ContextPacker:
        """创建 token-aware ContextPacker。"""

        return TokenAwareContextPacker(
            config=self.configs.build_context_packer_config(),
            token_estimator=self.build_token_estimator(),
        )

    def build_hybrid_retriever(
        self,
        *,
        vector_retriever: Retriever,
        bm25_retriever: Retriever,
    ) -> HybridRetriever:
        """使用现有的向量和 BM25 检索器创建 hybrid 检索器。"""

        config = self.configs.build_hybrid_retrieval_config()
        return HybridRetriever(
            sources=(
                HybridRetrievalSource(
                    name="vector",
                    retriever=vector_retriever,
                    weight=config.vector_weight,
                ),
                HybridRetrievalSource(
                    name="bm25",
                    retriever=bm25_retriever,
                    weight=config.bm25_weight,
                ),
            ),
            fusion_strategy=ReciprocalRankFusion(
                rank_constant=config.rrf_rank_constant
            ),
            config=config,
        )

    def build_retriever_registry(self, index: RagIndex) -> RetrieverRegistry:
        """为一个 RagIndex 创建内置检索策略的惰性注册表。"""

        registry = RetrieverRegistry()
        registry.register("vector", lambda: self.build_vector_retriever(index))
        registry.register("bm25", lambda: self.build_bm25_retriever(index))
        registry.register(
            "hybrid",
            lambda: self.build_hybrid_retriever(
                vector_retriever=registry.resolve("vector"),
                bm25_retriever=registry.resolve("bm25"),
            ),
        )
        return registry

    def build_retriever(
        self,
        index: RagIndex,
        *,
        registry: RetrieverRegistry | None = None,
    ) -> Retriever:
        """根据检索策略创建在线问答默认检索器。"""

        active_registry = (
            registry if registry is not None else self.build_retriever_registry(index)
        )
        strategy = self.configs.build_retrieval_config().strategy
        try:
            return active_registry.resolve(strategy)
        except ValueError as exc:
            raise AppError(
                ErrorCode.INVALID_CONFIG,
                str(exc),
            ) from exc

    def build_search_service(
        self,
        index: RagIndex,
        *,
        registry: RetrieverRegistry | None = None,
    ) -> SearchService:
        """创建只执行检索的 SearchService。"""

        active_registry = (
            registry if registry is not None else self.build_retriever_registry(index)
        )
        reranking_config = self.configs.build_reranking_config()
        return SearchService(
            registry=active_registry,
            config=self.configs.build_retrieval_config(),
            reranking_config=reranking_config,
            reranker=self.build_reranker(reranking_config),
            reporter=self.build_retrieval_reporter(index, active_registry),
        )

    def build_compare_search_service(
        self,
        index: RagIndex,
        *,
        registry: RetrieverRegistry | None = None,
    ) -> CompareSearchService:
        """创建多策略检索比较服务。"""

        active_registry = (
            registry if registry is not None else self.build_retriever_registry(index)
        )
        reranking_config = self.configs.build_reranking_config()
        return CompareSearchService(
            registry=active_registry,
            config=self.configs.build_retrieval_config(),
            reranking_config=reranking_config,
            reranker=self.build_reranker(reranking_config),
            reporter=self.build_retrieval_reporter(index, active_registry),
            comparison_reporter=self.build_retrieval_comparison_reporter(
                index,
                active_registry,
            ),
        )

    def build_retrieval_reporter(
        self,
        index: RagIndex,
        registry: RetrieverRegistry,
    ) -> RetrievalReporter:
        """根据索引、配置和已注册策略创建 retrieval reporter。"""

        reporter = RetrievalReporter(
            config=self.configs.build_retrieval_report_config(),
            runtime_snapshot=self.build_retrieval_runtime_snapshot(index, registry),
            writer=RetrievalReportWriter(),
        )
        reporter.prepare_output_directory()
        return reporter

    def build_retrieval_comparison_reporter(
        self,
        index: RagIndex,
        registry: RetrieverRegistry,
    ) -> RetrievalComparisonReporter:
        """根据索引、配置和已注册策略创建 compare search 聚合报告组件。"""

        reporter = RetrievalComparisonReporter(
            config=self.configs.build_retrieval_report_config(),
            runtime_snapshot=self.build_retrieval_runtime_snapshot(index, registry),
            writer=RetrievalComparisonReportWriter(),
        )
        reporter.prepare_output_directory()
        return reporter

    def build_retrieval_runtime_snapshot(
        self,
        index: RagIndex,
        registry: RetrieverRegistry,
    ) -> RetrievalRuntimeSnapshot:
        """构建单策略与比较报告共同使用的运行时快照。"""

        manifest = index.manifest
        retrieval_config = self.configs.build_retrieval_config()
        hybrid_config = self.configs.build_hybrid_retrieval_config()
        reranking_config = self.configs.build_reranking_config()
        return RetrievalRuntimeSnapshot(
            index=RetrievalIndexSnapshot(
                index_id=manifest.index_id,
                schema_version=manifest.schema_version,
                status=manifest.status,
                config_hash=manifest.config_hash,
                document_set_hash=manifest.document_set_hash,
                document_count=manifest.document_count,
                chunk_count=manifest.chunk_count,
                vector_count=manifest.vector_count,
                embedding_provider=manifest.embedding_provider,
                embedding_model=manifest.embedding_model,
                embedding_dimension=manifest.embedding_dimension,
                vector_repository_type=manifest.vector_repository_type,
                vector_collection_name=manifest.vector_collection_name,
                distance_metric=manifest.distance_metric,
            ),
            config=RetrievalConfigSnapshot(
                default_strategy=retrieval_config.strategy,
                default_top_k=retrieval_config.top_k,
                deduplicate_by_chunk_id=retrieval_config.deduplicate_by_chunk_id,
                tokenizer_strategy=self.configs.build_tokenizer_config().strategy,
                bm25_k1=retrieval_config.bm25.k1,
                bm25_b=retrieval_config.bm25.b,
                hybrid_candidate_multiplier=hybrid_config.candidate_multiplier,
                hybrid_rrf_rank_constant=hybrid_config.rrf_rank_constant,
                hybrid_vector_weight=hybrid_config.vector_weight,
                hybrid_bm25_weight=hybrid_config.bm25_weight,
                reranking_enabled=reranking_config.enabled,
                reranking_strategy=reranking_config.strategy,
                reranking_candidate_limit=reranking_config.candidate_limit,
                reranking_failure_mode=reranking_config.failure_mode,
                registered_strategies=registry.list_strategies(),
            ),
        )
