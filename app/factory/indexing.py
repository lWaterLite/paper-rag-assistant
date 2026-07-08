"""Indexing 相关对象组装。"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.errors import AppError, ErrorCode
from app.factory.configs import ConfigFactory
from app.factory.ingestion import IngestionFactory
from app.indexing.embedding_cache import (
    EmbeddingCache,
    FileEmbeddingCache,
    InMemoryEmbeddingCache,
)
from app.indexing.embeddings import (
    EmbeddingClient,
    MockEmbeddingClient,
    OpenAIEmbeddingClient,
)
from app.indexing.index_builder import IndexBuilder, RagIndex
from app.indexing.index_loader import validate_index_from_storage
from app.indexing.report import IndexBuildReportWriter
from app.indexing.vector_collection import InMemoryVectorCollection, VectorCollection
from app.ingest.chunking.collection import ChunkCollection, InMemoryChunkCollection
from app.ingest.chunking.registry import ChunkerRegistry
from app.ingest.chunking.report import ChunkingReportWriter
from app.ingest.document_collection import (
    DocumentCollection,
    InMemoryDocumentCollection,
)
from app.ingest.pipeline import IngestionPipeline, IngestionReportWriter
from app.repositories.chunk_repository import ChunkRepository, LocalJsonChunkRepository
from app.repositories.document_repository import (
    DocumentRepository,
    LocalJsonDocumentRepository,
)
from app.repositories.index_manifest_repository import IndexManifestRepository
from app.repositories.vector_repository import (
    LocalJsonVectorRepository,
    VectorRepository,
)


@dataclass(slots=True)
class IndexingFactory:
    """组装离线索引构建与已有索引加载相关对象。"""

    configs: ConfigFactory
    ingestion: IngestionFactory

    def build_embedding_client(self) -> EmbeddingClient:
        """根据配置创建 embedding client。"""

        config = self.configs.build_embedding_config()
        if config.provider == "mock":
            return MockEmbeddingClient(config)
        if config.provider == "openai":
            return OpenAIEmbeddingClient(config)
        raise ValueError(f"不支持的 embedding provider：{config.provider}")

    def build_embedding_cache(self) -> EmbeddingCache:
        """根据配置创建 embedding cache。"""

        vector_repository_config = self.configs.build_vector_repository_config()
        if (
            vector_repository_config.repository_type == "local_json"
            and vector_repository_config.persist
        ):
            return FileEmbeddingCache(vector_repository_config.embedding_cache_path)
        return InMemoryEmbeddingCache()

    @staticmethod
    def build_vector_collection() -> VectorCollection:
        """创建空的向量运行时集合。"""

        return InMemoryVectorCollection()

    def build_vector_repository(self) -> VectorRepository:
        """根据配置创建向量集合持久化 Repository。"""

        config = self.configs.build_vector_repository_config()
        if config.repository_type in {"memory", "local_json"}:
            return LocalJsonVectorRepository(config.vector_collection_path)
        raise ValueError(f"不支持的 vector repository 类型：{config.repository_type}")

    @staticmethod
    def build_document_collection() -> DocumentCollection:
        """创建文档运行时集合。"""

        return InMemoryDocumentCollection()

    @staticmethod
    def build_chunk_collection() -> ChunkCollection:
        """创建 chunk 运行时集合。"""

        return InMemoryChunkCollection()

    def build_document_repository(self) -> DocumentRepository:
        """根据配置创建文档集合持久化 Repository。"""

        config = self.configs.build_vector_repository_config()
        return LocalJsonDocumentRepository(config.document_collection_path)

    def build_chunk_repository(self) -> ChunkRepository:
        """根据配置创建 chunk 集合持久化 Repository。"""

        config = self.configs.build_vector_repository_config()
        return LocalJsonChunkRepository(config.chunk_collection_path)

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
        """创建离线索引构建器。

        这里允许测试或实验显式覆盖某些依赖；生产入口使用默认组装即可。
        """

        embedding_config = self.configs.build_embedding_config()
        vector_repository_config = self.configs.build_vector_repository_config()
        index_builder_config = self.configs.build_index_builder_config()
        return IndexBuilder(
            config=index_builder_config,
            embedding_config=embedding_config,
            vector_repository_config=vector_repository_config,
            ingestion_pipeline=ingestion_pipeline
            if ingestion_pipeline is not None
            else self.ingestion.build_ingestion_pipeline(),
            chunker=self.ingestion.build_configured_chunker(
                chunker_registry=chunker_registry
            ),
            embedding_client=embedding_client
            if embedding_client is not None
            else self.build_embedding_client(),
            embedding_cache=embedding_cache
            if embedding_cache is not None
            else self.build_embedding_cache(),
            vector_collection=vector_collection
            if vector_collection is not None
            else self.build_vector_collection(),
            document_collection=document_collection
            if document_collection is not None
            else self.build_document_collection(),
            chunk_collection=chunk_collection
            if chunk_collection is not None
            else self.build_chunk_collection(),
            vector_repository=vector_repository
            if vector_repository is not None
            else self.build_vector_repository(),
            document_repository=document_repository
            if document_repository is not None
            else self.build_document_repository(),
            chunk_repository=chunk_repository
            if chunk_repository is not None
            else self.build_chunk_repository(),
            manifest_repository=IndexManifestRepository(
                vector_repository_config.collection_dir, index_builder_config
            ),
            build_report_writer=IndexBuildReportWriter(),
            ingestion_report_writer=ingestion_report_writer
            if ingestion_report_writer is not None
            else IngestionReportWriter(),
            ingestion_report_config=self.configs.build_ingestion_report_config(),
            chunking_report_writer=chunking_report_writer
            if chunking_report_writer is not None
            else ChunkingReportWriter(),
            chunking_report_config=self.configs.build_chunking_report_config(),
        )

    def build_rag_index_from_storage(self) -> RagIndex:
        """从已有持久化索引加载在线 RAG 索引。"""

        vector_repository_config = self.configs.build_vector_repository_config()
        if (
            vector_repository_config.repository_type != "local_json"
            or not vector_repository_config.persist
        ):
            raise AppError(
                ErrorCode.INVALID_CONFIG,
                "加载已有索引要求 vector_repository.type='local_json' 且 persist=true；"
                "memory 或未持久化配置没有可恢复的索引产物",
            )

        embedding_config = self.configs.build_embedding_config()
        vector_repository = self.build_vector_repository()
        document_repository = self.build_document_repository()
        chunk_repository = self.build_chunk_repository()
        vector_collection = vector_repository.load()
        manifest = IndexManifestRepository(
            vector_repository_config.collection_dir,
            self.configs.build_index_builder_config(),
        ).read()
        validate_index_from_storage(
            manifest=manifest,
            embedding_config=embedding_config,
            vector_repository_config=vector_repository_config,
            vector_collection=vector_collection,
        )
        return RagIndex(
            vector_collection=vector_collection,
            document_collection=document_repository.load(),
            chunk_collection=chunk_repository.load(),
            embedding_client=self.build_embedding_client(),
            manifest=manifest,
        )
