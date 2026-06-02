"""检索器。"""

from __future__ import annotations

from app.indexing.embeddings import EmbeddingClient
from app.indexing.vector_store import InMemoryVectorStore
from app.core.models import RetrievedChunk


class VectorRetriever:
    """基于向量相似度的检索器。"""

    def __init__(self, embedding_client: EmbeddingClient, vector_store: InMemoryVectorStore) -> None:
        self._embedding_client = embedding_client
        self._vector_store = vector_store

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """检索与 query 最相似的 chunk。"""

        query_vector = self._embedding_client.embed_text(query)
        return self._vector_store.search(query_vector, top_k=top_k)

    # TODO 练习 9：
    # 当前只有 vector retrieval。
    # 请你预留 BM25Retriever 的接口，并思考它应该如何和 VectorRetriever 返回同一种 RetrievedChunk。

