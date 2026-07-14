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
from app.indexing.embedding_cache import EmbeddingCache
from app.indexing.embeddings import EmbeddingClient
from app.indexing.index_builder import IndexBuilder, RagIndex
from app.indexing.vector_collection import VectorCollection
from app.ingest.chunking.collection import ChunkCollection
from app.ingest.chunking.registry import ChunkerRegistry
from app.ingest.chunking.report import ChunkingReportWriter
from app.ingest.document_collection import DocumentCollection
from app.ingest.pipeline import IngestionPipeline, IngestionReportWriter
from app.pipeline import RagPipeline
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.vector_repository import VectorRepository
from app.retrieval.context import ContextPacker
from app.retrieval.retrievers import Retriever, RetrieverRegistry
from app.retrieval.services.search import CompareSearchService, SearchService
from app.retrieval.tokenizers import TokenizerRegistry
from app.retrieval.rerankers import RerankerRegistry
from app.retrieval.context.token_estimators import TokenEstimatorRegistry
from app.retrieval.context.evidence_transformers import EvidenceTransformerRegistry


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
            **(
                {"chunker_registry": self.chunker_registry}
                if self.chunker_registry is not None
                else {}
            ),
        )
        self.indexing = IndexingFactory(configs=self.configs, ingestion=self.ingestion)
        self.retrieval = RetrievalFactory(
            configs=self.configs,
            **(
                {"tokenizer_registry": self.tokenizer_registry}
                if self.tokenizer_registry is not None
                else {}
            ),
            **(
                {"reranker_registry": self.reranker_registry}
                if self.reranker_registry is not None
                else {}
            ),
            **(
                {"token_estimator_registry": self.token_estimator_registry}
                if self.token_estimator_registry is not None
                else {}
            ),
            **(
                {"evidence_transformer_registry": self.evidence_transformer_registry}
                if self.evidence_transformer_registry is not None
                else {}
            ),
        )
        self.pipelines = PipelineFactory(configs=self.configs, retrieval=self.retrieval)

    def build_index_builder(
        self,
        *,
        ingestion_pipeline: IngestionPipeline | None = None,
        embedding_client: EmbeddingClient | None = None,
        embedding_cache: EmbeddingCache | None = None,
        vector_collection: VectorCollection | None = None,
        document_collection: DocumentCollection | None = None,
        chunk_collection: ChunkCollection | None = None,
        vector_repository: VectorRepository | None = None,
        document_repository: DocumentRepository | None = None,
        chunk_repository: ChunkRepository | None = None,
        ingestion_report_writer: IngestionReportWriter | None = None,
        chunking_report_writer: ChunkingReportWriter | None = None,
        chunker_registry: ChunkerRegistry | None = None,
    ) -> IndexBuilder:
        """创建离线索引构建器。"""

        return self.indexing.build_index_builder(
            ingestion_pipeline=ingestion_pipeline,
            embedding_client=embedding_client,
            embedding_cache=embedding_cache,
            vector_collection=vector_collection,
            document_collection=document_collection,
            chunk_collection=chunk_collection,
            vector_repository=vector_repository,
            document_repository=document_repository,
            chunk_repository=chunk_repository,
            ingestion_report_writer=ingestion_report_writer,
            chunking_report_writer=chunking_report_writer,
            chunker_registry=chunker_registry,
        )

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
