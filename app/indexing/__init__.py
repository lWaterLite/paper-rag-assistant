"""索引构建、索引恢复与索引运行时组件。"""

from app.indexing.collections import (
    InMemoryVectorCollection,
    VectorCollection,
    VectorRecord,
    VectorSearchResult,
)
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
    InMemoryEmbeddingCache,
    build_default_embedding_client_registry,
)
from app.indexing.manifests import IndexManifest
from app.indexing.pipeline import IndexBuildResult, IndexBuilder, IndexLoader, RagIndex
from app.indexing.reporting import IndexBuildReportWriter

__all__ = [
    "EmbeddingCache",
    "EmbeddingClient",
    "EmbeddingClientRegistry",
    "EmbeddingConfig",
    "FileEmbeddingCache",
    "InMemoryEmbeddingCache",
    "InMemoryVectorCollection",
    "IndexBuildReportWriter",
    "IndexBuildResult",
    "IndexBuilder",
    "IndexBuilderConfig",
    "IndexLoader",
    "IndexManifest",
    "RagIndex",
    "VectorCollection",
    "VectorRecord",
    "VectorRepositoryConfig",
    "VectorSearchResult",
    "build_default_embedding_client_registry",
]
