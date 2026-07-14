"""应用级组合根。"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.settings import EnvSettings, ProjectSettings
from app.factory.configs import ConfigFactory
from app.factory.indexing import IndexingFactory
from app.factory.ingestion import IngestionFactory
from app.factory.pipelines import PipelineFactory
from app.factory.retrieval import RetrievalFactory
from app.generation.answer_generator import AnswerGenerator
from app.indexing.embeddings import (
    EmbeddingClientRegistry,
    build_default_embedding_client_registry,
)
from app.indexing.pipeline import IndexBuilder, RagIndex
from app.ingest.chunking.registry import ChunkerRegistry, build_default_chunker_registry
from app.pipeline import RagPipeline
from app.retrieval.context import ContextPacker
from app.retrieval.retrievers import Retriever, RetrieverRegistry
from app.retrieval.services.search import CompareSearchService, SearchService
from app.retrieval.tokenizers import TokenizerRegistry, build_default_tokenizer_registry
from app.retrieval.rerankers import RerankerRegistry
from app.retrieval.context.token_estimators import (
    TokenEstimatorRegistry,
    build_default_token_estimator_registry,
)
from app.retrieval.context.evidence_transformers import EvidenceTransformerRegistry
from app.retrieval.context.evidence_transformers import (
    build_default_evidence_transformer_registry,
)


@dataclass(slots=True)
class ApplicationFactory:
    """应用对象组合根。

    这个对象持有同一组 EnvSettings 和 ProjectSettings，并把它们传递给下级工厂。
    调用方应优先创建一个 ApplicationFactory，而不是在各处散落调用无状态构建函数。
    """

    env_settings: EnvSettings = field(default_factory=EnvSettings)
    project_settings: ProjectSettings = field(default_factory=ProjectSettings)
    chunker_registry: ChunkerRegistry | None = None
    tokenizer_registry: TokenizerRegistry | None = None
    reranker_registry: RerankerRegistry | None = None
    token_estimator_registry: TokenEstimatorRegistry | None = None
    evidence_transformer_registry: EvidenceTransformerRegistry | None = None
    embedding_registry: EmbeddingClientRegistry | None = None
    configs: ConfigFactory = field(init=False)
    ingestion: IngestionFactory = field(init=False)
    indexing: IndexingFactory = field(init=False)
    retrieval: RetrievalFactory = field(init=False)
    pipelines: PipelineFactory = field(init=False)

    def __post_init__(self) -> None:
        self.configs = ConfigFactory(
            env_settings=self.env_settings,
            project_settings=self.project_settings,
        )
        self.ingestion = IngestionFactory(
            configs=self.configs,
            chunker_registry=(
                self.chunker_registry
                if self.chunker_registry is not None
                else build_default_chunker_registry()
            ),
        )
        self.indexing = IndexingFactory(
            configs=self.configs,
            ingestion=self.ingestion,
            embedding_registry=(
                self.embedding_registry
                if self.embedding_registry is not None
                else build_default_embedding_client_registry()
            ),
        )
        self.retrieval = RetrievalFactory(
            configs=self.configs,
            tokenizer_registry=(
                self.tokenizer_registry
                if self.tokenizer_registry is not None
                else build_default_tokenizer_registry()
            ),
            reranker_registry=self.reranker_registry,
            token_estimator_registry=(
                self.token_estimator_registry
                if self.token_estimator_registry is not None
                else build_default_token_estimator_registry()
            ),
            evidence_transformer_registry=(
                self.evidence_transformer_registry
                if self.evidence_transformer_registry is not None
                else build_default_evidence_transformer_registry()
            ),
        )
        self.pipelines = PipelineFactory(configs=self.configs, retrieval=self.retrieval)

    def build_index_builder(
        self,
    ) -> IndexBuilder:
        """创建离线索引构建器。"""

        return self.indexing.build_index_builder()

    def build_rag_index_from_storage(self) -> RagIndex:
        """从已有持久化索引加载在线 RAG 索引。"""

        return self.indexing.build_rag_index_from_storage()

    def build_search_service(
        self,
        index: RagIndex,
        *,
        retriever_registry: RetrieverRegistry | None = None,
    ) -> SearchService:
        """创建只执行检索的 SearchService。"""

        return self.retrieval.build_search_service(
            index,
            registry=retriever_registry,
        )

    def build_compare_search_service(
        self,
        index: RagIndex,
        *,
        retriever_registry: RetrieverRegistry | None = None,
    ) -> CompareSearchService:
        """创建多策略检索比较服务。"""

        return self.retrieval.build_compare_search_service(
            index,
            registry=retriever_registry,
        )

    def build_rag_pipeline(
        self,
        index: RagIndex,
        *,
        retriever: Retriever | None = None,
        retriever_registry: RetrieverRegistry | None = None,
        context_packer: ContextPacker | None = None,
        answer_generator: AnswerGenerator | None = None,
    ) -> RagPipeline:
        """创建在线 RAG 问答 pipeline。"""

        return self.pipelines.build_rag_pipeline(
            index,
            retriever=retriever,
            retriever_registry=retriever_registry,
            context_packer=context_packer,
            answer_generator=answer_generator,
        )
