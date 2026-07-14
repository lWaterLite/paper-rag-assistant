"""已持久化索引与当前运行配置的兼容性校验。"""

from __future__ import annotations

from app.indexing.configuration import EmbeddingConfig, VectorRepositoryConfig
from app.indexing.manifests.models import (
    CURRENT_INDEX_SCHEMA_VERSION,
    READY_INDEX_STATUS,
    IndexManifest,
)


def validate_manifest_compatible(
    *,
    manifest: IndexManifest,
    embedding_config: EmbeddingConfig,
    vector_repository_config: VectorRepositoryConfig,
) -> None:
    """校验已有索引是否可由当前配置安全恢复。"""

    mismatches: list[str] = []
    if manifest.schema_version != CURRENT_INDEX_SCHEMA_VERSION:
        mismatches.append(
            f"schema_version: manifest={manifest.schema_version}, current={CURRENT_INDEX_SCHEMA_VERSION}"
        )
    if manifest.status != READY_INDEX_STATUS:
        mismatches.append(
            f"status: manifest={manifest.status}, required={READY_INDEX_STATUS}"
        )
    _append_mismatch(
        mismatches,
        "embedding_provider",
        manifest.embedding_provider,
        embedding_config.provider,
    )
    _append_mismatch(
        mismatches,
        "embedding_model",
        manifest.embedding_model,
        embedding_config.model,
    )
    _append_mismatch(
        mismatches,
        "embedding_dimension",
        manifest.embedding_dimension,
        embedding_config.dimension,
    )
    _append_mismatch(
        mismatches,
        "embedding_batch_size",
        manifest.embedding_batch_size,
        embedding_config.batch_size,
    )
    _append_mismatch(
        mismatches,
        "vector_repository_type",
        manifest.vector_repository_type,
        vector_repository_config.repository_type,
    )
    _append_mismatch(
        mismatches,
        "vector_collection_name",
        manifest.vector_collection_name,
        vector_repository_config.collection_name,
    )
    _append_mismatch(
        mismatches,
        "distance_metric",
        manifest.distance_metric,
        vector_repository_config.distance_metric,
    )
    if mismatches:
        raise ValueError("索引 manifest 与当前配置不兼容：" + "；".join(mismatches))


def _append_mismatch(
    mismatches: list[str],
    field: str,
    manifest_value: object,
    current_value: object,
) -> None:
    if manifest_value != current_value:
        mismatches.append(f"{field}: manifest={manifest_value}, current={current_value}")
