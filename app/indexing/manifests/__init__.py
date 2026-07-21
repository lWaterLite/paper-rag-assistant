"""索引版本 Manifest 与兼容性校验。"""

from app.indexing.manifests.compatibility import validate_manifest_compatible
from app.indexing.manifests.models import (
    BUILDING_INDEX_STATUS,
    CURRENT_INDEX_SCHEMA_VERSION,
    CURRENT_VECTOR_COLLECTION_SCHEMA_VERSION,
    FAILED_INDEX_STATUS,
    READY_INDEX_STATUS,
    EmbeddingRuntimeCompatibility,
    IndexArtifactDefinition,
    IndexBuildProvenance,
    IndexManifest,
    IndexRuntimeCompatibility,
    IndexStorageLocator,
    IndexVersionStatus,
    VectorCollectionRuntimeCompatibility,
)

__all__ = [
    "BUILDING_INDEX_STATUS",
    "CURRENT_INDEX_SCHEMA_VERSION",
    "CURRENT_VECTOR_COLLECTION_SCHEMA_VERSION",
    "EmbeddingRuntimeCompatibility",
    "FAILED_INDEX_STATUS",
    "IndexArtifactDefinition",
    "IndexBuildProvenance",
    "READY_INDEX_STATUS",
    "IndexManifest",
    "IndexRuntimeCompatibility",
    "IndexStorageLocator",
    "IndexVersionStatus",
    "VectorCollectionRuntimeCompatibility",
    "validate_manifest_compatible",
]
