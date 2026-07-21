"""已持久化索引与当前运行配置的兼容性校验。"""

from __future__ import annotations

from app.indexing.configuration import EmbeddingConfig, VectorRepositoryConfig
from app.indexing.manifests.models import (
    CURRENT_INDEX_SCHEMA_VERSION,
    CURRENT_VECTOR_COLLECTION_SCHEMA_VERSION,
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
    embedding = manifest.artifact_definition.runtime_compatibility.embedding
    vector_collection = (
        manifest.artifact_definition.runtime_compatibility.vector_collection
    )
    _append_mismatch(
        mismatches,
        "embedding_provider",
        embedding.provider,
        embedding_config.provider,
    )
    _append_mismatch(
        mismatches,
        "embedding_model",
        embedding.model,
        embedding_config.model,
    )
    _append_mismatch(
        mismatches,
        "embedding_dimension",
        embedding.dimension,
        embedding_config.dimension,
    )
    _append_mismatch(
        mismatches,
        "vector_repository_type",
        vector_collection.repository_type,
        vector_repository_config.repository_type,
    )
    _append_mismatch(
        mismatches,
        "distance_metric",
        vector_collection.distance_metric,
        vector_repository_config.distance_metric,
    )
    _append_mismatch(
        mismatches,
        "vector_collection_schema_version",
        vector_collection.schema_version,
        CURRENT_VECTOR_COLLECTION_SCHEMA_VERSION,
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
