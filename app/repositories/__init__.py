"""持久化 Repository 实现。"""

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

__all__ = [
    "ChunkRepository",
    "DocumentRepository",
    "IndexManifestRepository",
    "LocalJsonChunkRepository",
    "LocalJsonDocumentRepository",
    "LocalJsonVectorRepository",
    "VectorRepository",
]
