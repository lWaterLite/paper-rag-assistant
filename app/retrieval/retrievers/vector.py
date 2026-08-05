"""向量检索器。"""

from __future__ import annotations

from app.core.errors import AppError, ErrorCode
from app.indexing.collections import VectorCollection
from app.indexing.embeddings.base import EmbeddingClient
from app.ingest.chunking.collection import ChunkCollection
from app.retrieval.models import RetrievedChunk
from app.retrieval.retrievers.result_builder import RetrievedChunkBuilder


class VectorRetriever:
    """基于向量相似度的检索器。"""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        vector_collection: VectorCollection,
        chunk_collection: ChunkCollection,
    ) -> None:
        self._embedding_client = embedding_client
        self._vector_collection = vector_collection
        self._chunk_collection = chunk_collection

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """检索与 query 最相似的 chunk。"""

        if top_k <= 0 or not query.strip():
            return []

        query_vector = self._embedding_client.embed_text(query)
        vector_results = self._vector_collection.search(query_vector, top_k=top_k)
        retrieved_chunks: list[RetrievedChunk] = []

        for result in vector_results:
            chunk = self._chunk_collection.get_by_id(result.chunk_id)
            if chunk is None:
                raise AppError(
                    ErrorCode.RETRIEVAL_FAILED,
                    f"向量命中了 chunk_id={result.chunk_id}，但 chunk 集合中找不到对应内容",
                )
            retrieved_chunks.append(
                RetrievedChunkBuilder.from_chunk(
                    chunk,
                    score=result.score,
                    rank=result.rank,
                    retriever="vector",
                )
            )

        return retrieved_chunks
