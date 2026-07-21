"""Indexing 相关对象组装。"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.factory.configs import ConfigFactory
from app.factory.ingestion import IngestionFactory
from app.indexing.collections import InMemoryVectorCollection, VectorCollection
from app.indexing.configuration import (
    EmbeddingConfig,
    IndexBuilderConfig,
    VectorRepositoryConfig,
)
from app.indexing.embeddings import (
    EmbeddingCache,
    EmbeddingClient,
    EmbeddingClientRegistry,
    FileEmbeddingCache,
    build_default_embedding_client_registry,
)
from app.indexing.pipeline import IndexBuilder, IndexLoader, RagIndex
from app.indexing.reporting import IndexBuildReportWriter
from app.ingest.chunking.collection import ChunkCollection, InMemoryChunkCollection
from app.ingest.collections import (
    DocumentCollection,
    InMemoryDocumentCollection,
)
from app.repositories.chunk import ChunkRepository
from app.repositories.document import DocumentRepository
from app.repositories.manifest import ManifestRepository
from app.repositories.registries import (
    ChunkRepositoryRegistry,
    DocumentRepositoryRegistry,
    ManifestRepositoryRegistry,
    VectorRepositoryRegistry,
    build_default_chunk_repository_registry,
    build_default_document_repository_registry,
    build_default_manifest_repository_registry,
    build_default_vector_repository_registry,
)
from app.repositories.vector import VectorRepository


@dataclass(slots=True)
class IndexingFactory:
    """组装离线索引构建与已有索引加载相关对象。"""

    configs: ConfigFactory
    ingestion: IngestionFactory
    embedding_registry: EmbeddingClientRegistry = field(
        default_factory=build_default_embedding_client_registry
    )
    vector_repository_registry: VectorRepositoryRegistry = field(
        default_factory=build_default_vector_repository_registry
    )
    document_repository_registry: DocumentRepositoryRegistry = field(
        default_factory=build_default_document_repository_registry
    )
    chunk_repository_registry: ChunkRepositoryRegistry = field(
        default_factory=build_default_chunk_repository_registry
    )
    manifest_repository_registry: ManifestRepositoryRegistry = field(
        default_factory=build_default_manifest_repository_registry
    )
    embedding_cache: EmbeddingCache | None = None

    def _build_embedding_client(self, config: EmbeddingConfig) -> EmbeddingClient:
        """通过注册表创建当前配置指定的 embedding 客户端。"""

        secret = self.configs.env_settings.openai_api_key
        return self.embedding_registry.create(
            config,
            api_key=secret.get_secret_value() if secret is not None else None,
        )

    def _build_embedding_cache(
        self,
        vector_repository_config: VectorRepositoryConfig,
    ) -> EmbeddingCache:
        """创建当前索引构建使用的 embedding 缓存。"""

        if self.embedding_cache is not None:
            return self.embedding_cache

        return FileEmbeddingCache(vector_repository_config.embedding_cache_path)

    @staticmethod
    def _build_vector_collection() -> VectorCollection:
        """创建空的向量运行时集合。"""

        return InMemoryVectorCollection()

    def _build_vector_repository(
        self,
        config: VectorRepositoryConfig,
    ) -> VectorRepository:
        """根据配置创建向量集合持久化 Repository。"""

        return self.vector_repository_registry.create(
            config.repository_type,
            path=config.vector_collection_path,
        )

    @staticmethod
    def _build_document_collection() -> DocumentCollection:
        """创建文档运行时集合。"""

        return InMemoryDocumentCollection()

    @staticmethod
    def _build_chunk_collection() -> ChunkCollection:
        """创建 chunk 运行时集合。"""

        return InMemoryChunkCollection()

    def _build_document_repository(
        self,
        config: VectorRepositoryConfig,
    ) -> DocumentRepository:
        """根据配置创建文档集合持久化 Repository。"""

        return self.document_repository_registry.create(
            config.repository_type,
            path=config.document_collection_path,
        )

    def _build_chunk_repository(
        self,
        config: VectorRepositoryConfig,
    ) -> ChunkRepository:
        """根据配置创建 chunk 集合持久化 Repository。"""

        return self.chunk_repository_registry.create(
            config.repository_type,
            path=config.chunk_collection_path,
        )

    def _build_manifest_repository(
        self,
        *,
        vector_repository_config: VectorRepositoryConfig,
        index_builder_config: IndexBuilderConfig,
    ) -> ManifestRepository:
        """根据配置创建索引 Manifest 持久化 Repository。"""

        return self.manifest_repository_registry.create(
            vector_repository_config.repository_type,
            index_dir=vector_repository_config.collection_dir,
            config=index_builder_config,
        )

    def build_index_builder(
        self,
    ) -> IndexBuilder:
        """组装离线索引构建流程。"""

        ingestion_dependencies = self.ingestion.build_indexing_dependencies()
        config = self.configs.indexing
        embedding_client = self._build_embedding_client(config.embedding)
        return IndexBuilder(
            config=config.index_builder,
            embedding_config=config.embedding,
            vector_repository_config=config.vector_repository,
            ingestion_pipeline=ingestion_dependencies.pipeline,
            chunker=ingestion_dependencies.chunker,
            embedding_client=embedding_client,
            embedding_cache=self._build_embedding_cache(config.vector_repository),
            vector_collection=self._build_vector_collection(),
            document_collection=self._build_document_collection(),
            chunk_collection=self._build_chunk_collection(),
            vector_repository=self._build_vector_repository(config.vector_repository),
            document_repository=self._build_document_repository(config.vector_repository),
            chunk_repository=self._build_chunk_repository(config.vector_repository),
            manifest_repository=self._build_manifest_repository(
                vector_repository_config=config.vector_repository,
                index_builder_config=config.index_builder,
            ),
            build_report_writer=IndexBuildReportWriter(),
            ingestion_report_writer=ingestion_dependencies.ingestion_report_writer,
            ingestion_report_config=ingestion_dependencies.ingestion_report_config,
            chunking_report_writer=ingestion_dependencies.chunking_report_writer,
            chunking_report_config=ingestion_dependencies.chunking_report_config,
        )

    def build_rag_index_from_storage(self) -> RagIndex:
        """从已有持久化索引加载在线 RAG 索引。"""

        config = self.configs.indexing
        loader = IndexLoader(
            embedding_config=config.embedding,
            vector_repository_config=config.vector_repository,
            embedding_client=self._build_embedding_client(config.embedding),
            vector_repository=self._build_vector_repository(config.vector_repository),
            document_repository=self._build_document_repository(config.vector_repository),
            chunk_repository=self._build_chunk_repository(config.vector_repository),
            manifest_repository=self._build_manifest_repository(
                vector_repository_config=config.vector_repository,
                index_builder_config=config.index_builder,
            ),
        )
        return loader.load()
