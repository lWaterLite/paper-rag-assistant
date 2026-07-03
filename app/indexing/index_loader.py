"""已有索引加载校验辅助。"""

from __future__ import annotations

from app.core.errors import AppError, ErrorCode
from app.indexing.configs import EmbeddingConfig, VectorStoreConfig
from app.indexing.manifest import IndexManifest, validate_manifest_compatible
from app.indexing.vector_store import VectorStore


def validate_index_from_storage(
    *,
    manifest: IndexManifest,
    embedding_config: EmbeddingConfig,
    vector_store_config: VectorStoreConfig,
    vector_store: VectorStore,
) -> None:
    """校验已有索引是否可以被当前配置安全加载。"""

    try:
        validate_manifest_compatible(
            manifest=manifest,
            embedding_config=embedding_config,
            vector_store_config=vector_store_config,
        )
    except ValueError as exc:
        raise AppError(ErrorCode.INVALID_CONFIG, str(exc)) from exc

    if vector_store.count() != manifest.vector_count:
        raise AppError(
            ErrorCode.INDEX_FAILED,
            f"索引向量数量与 manifest 不一致：manifest={manifest.vector_count}，"
            f"vector_store={vector_store.count()}",
        )

    if manifest.vector_count > 0 and vector_store.dimension != manifest.embedding_dimension:
        raise AppError(
            ErrorCode.INDEX_FAILED,
            f"索引向量维度与 manifest 不一致：manifest={manifest.embedding_dimension}，"
            f"vector_store={vector_store.dimension}",
        )
