"""可替换 Embedding Provider 的扩展契约。"""

from app.indexing.embeddings.base import EmbeddingClient
from app.indexing.embeddings.registry import EmbeddingClientRegistry

__all__ = [
    "EmbeddingClient",
    "EmbeddingClientRegistry",
]
