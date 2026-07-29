"""Retrieval 相关对象组装。"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.errors import AppError, ErrorCode
from app.factory.configs import ConfigFactory
from app.indexing.pipeline.types import RagIndex
from app.retrieval.configuration import HybridRetrievalConfig, RetrievalConfig
from app.retrieval.configuration.postprocessing import PostProcessingConfig
from app.retrieval.configuration.postprocessing.profile import PostProcessingProfile
from app.retrieval.configuration.postprocessing.validator import (
    PostProcessingConfigValidator,
)
from app.retrieval.context import ContextPacker
from app.retrieval.context.evidence_transformers import (
    EvidenceTransformationConfig,
    EvidenceTransformer,
    EvidenceTransformerRegistry,
)
from app.retrieval.context.evidence_transformers.registry import (
    build_default_evidence_transformer_registry,
)
from app.retrieval.context.evidence_transformers.stage import EvidenceTransformStage
from app.retrieval.context.packer import TokenAwareContextPacker
from app.retrieval.context.token_estimators import (
    TokenEstimator,
    TokenEstimatorConfig,
    TokenEstimatorRegistry,
)
from app.retrieval.context.token_estimators.registry import (
    build_default_token_estimator_registry,
)
from app.retrieval.reporting.comparison_reporter import RetrievalComparisonReporter
from app.retrieval.reporting.comparison_writer import RetrievalComparisonReportWriter
from app.retrieval.reporting.models import (
    RetrievalConfigSnapshot,
    RetrievalIndexSnapshot,
    RetrievalRuntimeSnapshot,
)
from app.retrieval.reporting.reporter import RetrievalReporter
from app.retrieval.reporting.writer import RetrievalReportWriter
from app.retrieval.rerankers import (
    Reranker,
    RerankerRegistry,
    RerankingConfig,
)
from app.retrieval.rerankers.registry import build_default_reranker_registry
from app.retrieval.retrievers import Retriever, RetrieverRegistry
from app.retrieval.retrievers.bm25 import BM25Index, BM25Retriever
from app.retrieval.retrievers.fusion.rrf import ReciprocalRankFusion
from app.retrieval.retrievers.hybrid import HybridRetrievalSource, HybridRetriever
from app.retrieval.retrievers.vector import VectorRetriever
from app.retrieval.services.search import CompareSearchService, SearchService
from app.retrieval.tokenizers import Tokenizer, TokenizerConfig, TokenizerRegistry
from app.retrieval.tokenizers.registry import build_default_tokenizer_registry


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
    evidence_transformer_registry: EvidenceTransformerRegistry = field(
        default_factory=build_default_evidence_transformer_registry
    )

    @staticmethod
    def build_vector_retriever(index: RagIndex) -> VectorRetriever:
        """创建向量检索器。"""

        return VectorRetriever(
            index.embedding_client,
            index.vector_collection,
            index.chunk_collection,
        )

    def build_bm25_retriever(
        self,
        index: RagIndex,
        *,
        retrieval_config: RetrievalConfig | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> BM25Retriever:
        """根据当前 chunk collection 创建 BM25 检索器。"""

        active_config = (
            retrieval_config
            if retrieval_config is not None
            else self.configs.retrieval.retrieval
        )
        bm25_index = BM25Index.from_chunks(
            index.chunk_collection.iter_chunks(),
            config=active_config.bm25,
            tokenizer=(
                tokenizer
                if tokenizer is not None
                else self.build_tokenizer(config=self.configs.retrieval.tokenizer)
            ),
        )
        return BM25Retriever(bm25_index)

    def build_tokenizer(
        self,
        *,
        config: TokenizerConfig | None = None,
    ) -> Tokenizer:
        """根据当前配置创建 BM25 使用的分词器。"""

        active_config = (
            config if config is not None else self.configs.retrieval.tokenizer
        )
        return self.tokenizer_registry.create(active_config)

    def build_reranker(
        self,
        config: RerankingConfig | None = None,
    ) -> Reranker | None:
        """根据配置创建 reranker；禁用时显式返回 None。"""

        active_config = (
            config if config is not None else self.configs.retrieval.reranking
        )
        if not active_config.enabled:
            return None
        active_registry = (
            self.reranker_registry
            if self.reranker_registry is not None
            else build_default_reranker_registry(
                self.build_tokenizer(config=self.configs.retrieval.tokenizer)
            )
        )
        try:
            return active_registry.create(active_config)
        except ValueError as exc:
            raise AppError(ErrorCode.INVALID_CONFIG, str(exc)) from exc

    def build_token_estimator(
        self,
        *,
        config: TokenEstimatorConfig | None = None,
    ) -> TokenEstimator:
        """根据配置创建生成模型窗口使用的 token estimator。"""

        try:
            active_config = (
                config if config is not None else self.configs.retrieval.token_estimator
            )
            return self.token_estimator_registry.create(active_config)
        except ValueError as exc:
            raise AppError(ErrorCode.INVALID_CONFIG, str(exc)) from exc

    def build_context_packer(
        self,
        *,
        postprocessing_config: PostProcessingConfig | None = None,
        token_estimator_config: TokenEstimatorConfig | None = None,
    ) -> ContextPacker:
        """创建 token-aware ContextPacker。"""

        active_config = self._resolve_postprocessing_config(postprocessing_config)
        return TokenAwareContextPacker(
            config=active_config.context_packing,
            token_estimator=self.build_token_estimator(config=token_estimator_config),
        )

    def build_evidence_transformer(
        self,
        config: EvidenceTransformationConfig | None = None,
    ) -> EvidenceTransformer | None:
        """根据配置创建候选证据变换器；禁用时显式返回 None。"""

        active_config = (
            config
            if config is not None
            else self.configs.retrieval.evidence_transformation
        )
        if not active_config.enabled:
            return None
        try:
            return self.evidence_transformer_registry.create(active_config)
        except (TypeError, ValueError) as exc:
            raise AppError(ErrorCode.INVALID_CONFIG, str(exc)) from exc

    def build_evidence_transform_stage(
        self,
        *,
        postprocessing_config: PostProcessingConfig | None = None,
    ) -> EvidenceTransformStage:
        """创建仅供 RAG 上下文分支使用的候选证据变换阶段。"""

        active_config = self._resolve_postprocessing_config(postprocessing_config)
        transformation_config = active_config.evidence_transformation
        return EvidenceTransformStage(
            config=transformation_config,
            transformer=self.build_evidence_transformer(transformation_config),
        )

    def build_postprocessing_config(self) -> PostProcessingConfig:
        """构建并校验检索后处理流程的组合配置。"""

        config = self.configs.retrieval.postprocessing
        self._validate_postprocessing_config(config)
        return config

    @staticmethod
    def _validate_postprocessing_config(config: PostProcessingConfig) -> None:
        """把组合校验错误转换为应用统一错误。"""

        try:
            PostProcessingConfigValidator.validate(config)
        except ValueError as exc:
            raise AppError(ErrorCode.INVALID_CONFIG, str(exc)) from exc

    def _resolve_postprocessing_config(
        self,
        config: PostProcessingConfig | None,
    ) -> PostProcessingConfig:
        """解析调用方注入或当前 Factory 创建的后处理配置。"""

        if config is None:
            return self.build_postprocessing_config()
        self._validate_postprocessing_config(config)
        return config

    def build_postprocessing_profile(
        self,
        config: PostProcessingConfig | None = None,
    ) -> PostProcessingProfile:
        """构建供报告与诊断使用的后处理运行时摘要。"""

        active_config = self._resolve_postprocessing_config(config)
        return PostProcessingProfile.from_config(active_config)

    def build_hybrid_retriever(
        self,
        *,
        vector_retriever: Retriever,
        bm25_retriever: Retriever,
        config: HybridRetrievalConfig | None = None,
    ) -> HybridRetriever:
        """使用现有的向量和 BM25 检索器创建 hybrid 检索器。"""

        active_config = config if config is not None else self.configs.retrieval.hybrid
        return HybridRetriever(
            sources=(
                HybridRetrievalSource(
                    name="vector",
                    retriever=vector_retriever,
                    weight=active_config.vector_weight,
                ),
                HybridRetrievalSource(
                    name="bm25",
                    retriever=bm25_retriever,
                    weight=active_config.bm25_weight,
                ),
            ),
            fusion_strategy=ReciprocalRankFusion(
                rank_constant=active_config.rrf_rank_constant
            ),
            config=active_config,
        )

    def build_retriever_registry(self, index: RagIndex) -> RetrieverRegistry:
        """为一个 RagIndex 创建内置检索策略的惰性注册表。"""

        config = self.configs.retrieval
        tokenizer = self.build_tokenizer(config=config.tokenizer)
        registry = RetrieverRegistry()
        registry.register("vector", lambda: self.build_vector_retriever(index))
        registry.register(
            "bm25",
            lambda: self.build_bm25_retriever(
                index,
                retrieval_config=config.retrieval,
                tokenizer=tokenizer,
            ),
        )
        registry.register(
            "hybrid",
            lambda: self.build_hybrid_retriever(
                vector_retriever=registry.resolve("vector"),
                bm25_retriever=registry.resolve("bm25"),
                config=config.hybrid,
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
        strategy = self.configs.retrieval.retrieval.strategy
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
        postprocessing_config: PostProcessingConfig | None = None,
    ) -> SearchService:
        """创建只执行检索的 SearchService。"""

        active_registry = (
            registry if registry is not None else self.build_retriever_registry(index)
        )
        active_postprocessing_config = self._resolve_postprocessing_config(
            postprocessing_config
        )
        reranking_config = active_postprocessing_config.reranking
        return SearchService(
            registry=active_registry,
            config=active_postprocessing_config.retrieval,
            reranking_config=reranking_config,
            reranker=self.build_reranker(reranking_config),
            reporter=self.build_retrieval_reporter(
                index,
                active_registry,
                postprocessing_profile=self.build_postprocessing_profile(
                    active_postprocessing_config
                ),
            ),
        )

    def build_compare_search_service(
        self,
        index: RagIndex,
        *,
        registry: RetrieverRegistry | None = None,
        postprocessing_config: PostProcessingConfig | None = None,
    ) -> CompareSearchService:
        """创建多策略检索比较服务。"""

        active_registry = (
            registry if registry is not None else self.build_retriever_registry(index)
        )
        active_postprocessing_config = self._resolve_postprocessing_config(
            postprocessing_config
        )
        reranking_config = active_postprocessing_config.reranking
        postprocessing_profile = self.build_postprocessing_profile(
            active_postprocessing_config
        )
        return CompareSearchService(
            registry=active_registry,
            config=active_postprocessing_config.retrieval,
            reranking_config=reranking_config,
            reranker=self.build_reranker(reranking_config),
            reporter=self.build_retrieval_reporter(
                index,
                active_registry,
                postprocessing_profile=postprocessing_profile,
            ),
            comparison_reporter=self.build_retrieval_comparison_reporter(
                index,
                active_registry,
                postprocessing_profile=postprocessing_profile,
            ),
        )

    def build_retrieval_reporter(
        self,
        index: RagIndex,
        registry: RetrieverRegistry,
        *,
        postprocessing_profile: PostProcessingProfile,
    ) -> RetrievalReporter:
        """根据索引、配置和已注册策略创建 retrieval reporter。"""

        reporter = RetrievalReporter(
            config=self.configs.retrieval.report,
            runtime_snapshot=self.build_retrieval_runtime_snapshot(
                index,
                registry,
                postprocessing_profile=postprocessing_profile,
            ),
            writer=RetrievalReportWriter(),
        )
        reporter.prepare_output_directory()
        return reporter

    def build_retrieval_comparison_reporter(
        self,
        index: RagIndex,
        registry: RetrieverRegistry,
        *,
        postprocessing_profile: PostProcessingProfile,
    ) -> RetrievalComparisonReporter:
        """根据索引、配置和已注册策略创建 compare search 聚合报告组件。"""

        reporter = RetrievalComparisonReporter(
            config=self.configs.retrieval.report,
            runtime_snapshot=self.build_retrieval_runtime_snapshot(
                index,
                registry,
                postprocessing_profile=postprocessing_profile,
            ),
            writer=RetrievalComparisonReportWriter(),
        )
        reporter.prepare_output_directory()
        return reporter

    def build_retrieval_runtime_snapshot(
        self,
        index: RagIndex,
        registry: RetrieverRegistry,
        *,
        postprocessing_profile: PostProcessingProfile,
    ) -> RetrievalRuntimeSnapshot:
        """构建单策略与比较报告共同使用的运行时快照。"""

        manifest = index.manifest
        artifact_definition = manifest.artifact_definition
        runtime_compatibility = artifact_definition.runtime_compatibility
        config = self.configs.retrieval
        retrieval_config = config.retrieval
        hybrid_config = config.hybrid
        return RetrievalRuntimeSnapshot(
            index=RetrievalIndexSnapshot(
                index_id=manifest.index_id,
                schema_version=manifest.schema_version,
                status=manifest.status,
                artifact_definition_hash=manifest.artifact_definition_hash,
                document_set_hash=manifest.document_set_hash,
                document_count=manifest.document_count,
                chunk_count=manifest.chunk_count,
                vector_count=manifest.vector_count,
                embedding_provider=runtime_compatibility.embedding.provider,
                embedding_model=runtime_compatibility.embedding.model,
                embedding_dimension=runtime_compatibility.embedding.dimension,
                vector_repository_type=(
                    runtime_compatibility.vector_collection.repository_type
                ),
                vector_collection_name=manifest.storage_locator.collection_name,
                distance_metric=runtime_compatibility.vector_collection.distance_metric,
            ),
            config=RetrievalConfigSnapshot(
                default_strategy=retrieval_config.strategy,
                default_top_k=retrieval_config.top_k,
                deduplicate_by_chunk_id=retrieval_config.deduplicate_by_chunk_id,
                tokenizer_strategy=config.tokenizer.strategy,
                bm25_k1=retrieval_config.bm25.k1,
                bm25_b=retrieval_config.bm25.b,
                hybrid_candidate_multiplier=hybrid_config.candidate_multiplier,
                hybrid_rrf_rank_constant=hybrid_config.rrf_rank_constant,
                hybrid_vector_weight=hybrid_config.vector_weight,
                hybrid_bm25_weight=hybrid_config.bm25_weight,
                postprocessing=postprocessing_profile,
                registered_strategies=registry.list_strategies(),
            ),
        )
