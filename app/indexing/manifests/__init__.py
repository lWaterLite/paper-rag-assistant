"""索引版本 Manifest 与兼容性校验。"""

from app.indexing.manifests.compatibility import validate_manifest_compatible
from app.indexing.manifests.models import (
    BUILDING_INDEX_STATUS,
    CURRENT_INDEX_SCHEMA_VERSION,
    FAILED_INDEX_STATUS,
    READY_INDEX_STATUS,
    IndexManifest,
    IndexVersionStatus,
)

__all__ = [
    "BUILDING_INDEX_STATUS",
    "CURRENT_INDEX_SCHEMA_VERSION",
    "FAILED_INDEX_STATUS",
    "READY_INDEX_STATUS",
    "IndexManifest",
    "IndexVersionStatus",
    "validate_manifest_compatible",
]
