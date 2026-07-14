"""持久化索引恢复流程。"""

from __future__ import annotations

from app.core.errors import AppError, ErrorCode
from app.indexing.configuration import EmbeddingConfig, VectorRepositoryConfig
from app.indexing.embeddings.base import EmbeddingClient
from app.indexing.manifests.compatibility import validate_manifest_compatible
from app.indexing.pipeline.types import RagIndex
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.index_manifest_repository import IndexManifestRepository
from app.repositories.vector_repository import VectorRepository


class IndexLoader:
    """从 Repository 恢复并校验可供在线使用的 RAG 索引。"""

    def __init__(
        self,
        *,
        embedding_config: EmbeddingConfig,
        vector_repository_config: VectorRepositoryConfig,
        embedding_client: EmbeddingClient,
        vector_repository: VectorRepository,
        document_repository: DocumentRepository,
        chunk_repository: ChunkRepository,
        manifest_repository: IndexManifestRepository,
    ) -> None:
        self._embedding_config = embedding_config
        self._vector_repository_config = vector_repository_config
        self._embedding_client = embedding_client
        self._vector_repository = vector_repository
        self._document_repository = document_repository
        self._chunk_repository = chunk_repository
        self._manifest_repository = manifest_repository

    def load(self) -> RagIndex:
        """加载索引产物，校验兼容性后构造运行时索引。"""

        self._validate_loadable_configuration()
        vector_collection = self._vector_repository.load()
        manifest = self._manifest_repository.read()
        self._validate_loaded_artifacts(manifest=manifest, vector_collection=vector_collection)
        return RagIndex(
            vector_collection=vector_collection,
            document_collection=self._document_repository.load(),
            chunk_collection=self._chunk_repository.load(),
            embedding_client=self._embedding_client,
            manifest=manifest,
        )

    def _validate_loadable_configuration(self) -> None:
        """确认当前配置确实指向可恢复的持久化索引。"""

        config = self._vector_repository_config
        if config.repository_type != "local_json" or not config.persist:
            raise AppError(
                ErrorCode.INVALID_CONFIG,
                "加载已有索引要求 vector_repository.type='local_json' 且 persist=true；"
                "memory 或未持久化配置没有可恢复的索引产物",
            )

    def _validate_loaded_artifacts(self, *, manifest, vector_collection) -> None:
        """校验 Manifest 与已加载向量集合是否一致。"""

        try:
            validate_manifest_compatible(
                manifest=manifest,
                embedding_config=self._embedding_config,
                vector_repository_config=self._vector_repository_config,
            )
        except ValueError as exc:
            raise AppError(ErrorCode.INVALID_CONFIG, str(exc)) from exc

        if vector_collection.count() != manifest.vector_count:
            raise AppError(
                ErrorCode.INDEX_FAILED,
                f"索引向量数量与 manifest 不一致：manifest={manifest.vector_count}，"
                f"vector_collection={vector_collection.count()}",
            )
        if (
            manifest.vector_count > 0
            and vector_collection.dimension != manifest.embedding_dimension
        ):
            raise AppError(
                ErrorCode.INDEX_FAILED,
                f"索引向量维度与 manifest 不一致：manifest={manifest.embedding_dimension}，"
                f"vector_collection={vector_collection.dimension}",
            )
