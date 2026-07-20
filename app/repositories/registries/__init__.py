"""索引持久化 Repository 的注册与构建入口。"""

from app.repositories.registries.chunk import (
    ChunkRepositoryRegistry,
    build_default_chunk_repository_registry,
)
from app.repositories.registries.document import (
    DocumentRepositoryRegistry,
    build_default_document_repository_registry,
)
from app.repositories.registries.manifest import (
    ManifestRepositoryRegistry,
    build_default_manifest_repository_registry,
)
from app.repositories.index_manifest_repository import ManifestRepository
from app.repositories.registries.vector import (
    VectorRepositoryRegistry,
    build_default_vector_repository_registry,
)

__all__ = [
    "ChunkRepositoryRegistry",
    "DocumentRepositoryRegistry",
    "ManifestRepository",
    "ManifestRepositoryRegistry",
    "VectorRepositoryRegistry",
    "build_default_chunk_repository_registry",
    "build_default_document_repository_registry",
    "build_default_manifest_repository_registry",
    "build_default_vector_repository_registry",
]
