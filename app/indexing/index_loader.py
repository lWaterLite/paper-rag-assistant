"""已有索引加载校验辅助。"""

from __future__ import annotations

from app.core.errors import AppError, ErrorCode
from app.indexing.configs import EmbeddingConfig, VectorRepositoryConfig
from app.indexing.manifest import IndexManifest, validate_manifest_compatible
from app.indexing.vector_collection import VectorCollection


def validate_index_from_storage(
    *,
    manifest: IndexManifest,
    embedding_config: EmbeddingConfig,
    vector_repository_config: VectorRepositoryConfig,
    vector_collection: VectorCollection,
) -> None:
    """校验已有索引是否可以被当前配置安全加载。"""

    try:
        validate_manifest_compatible(
            manifest=manifest,
            embedding_config=embedding_config,
            vector_repository_config=vector_repository_config,
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
