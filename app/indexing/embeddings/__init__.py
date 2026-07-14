"""Embedding 客户端、缓存与 Provider 注册能力。"""

from app.indexing.embeddings.base import EmbeddingClient
from app.indexing.embeddings.cache import (
    EmbeddingCache,
    EmbeddingCacheKey,
    FileEmbeddingCache,
    InMemoryEmbeddingCache,
)
from app.indexing.embeddings.mock import MockEmbeddingClient
from app.indexing.embeddings.openai import OpenAIEmbeddingClient
from app.indexing.embeddings.registry import (
    EmbeddingClientRegistry,
    build_default_embedding_client_registry,
)
from app.indexing.embeddings.validation import validate_embedding_vectors

__all__ = [
    "EmbeddingCache",
    "EmbeddingCacheKey",
    "EmbeddingClient",
    "EmbeddingClientRegistry",
    "FileEmbeddingCache",
    "InMemoryEmbeddingCache",
    "MockEmbeddingClient",
    "OpenAIEmbeddingClient",
    "build_default_embedding_client_registry",
    "validate_embedding_vectors",
]
