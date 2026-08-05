"""检索器命中结果组装。"""

from __future__ import annotations

from app.ingest.chunking.models import DocumentChunk
from app.retrieval.models import RetrievedChunk


class RetrievedChunkBuilder:
    """把底层检索命中转换成统一的 RetrievedChunk。

    BM25 和向量检索的内部命中结构不同，但在线检索层向外暴露的结果应该保持一致。
    这个 builder 收束字段映射逻辑，避免每个检索器重复拼装 RetrievedChunk。
    """

    @staticmethod
    def from_chunk(
        chunk: DocumentChunk,
        *,
        score: float,
        rank: int,
        retriever: str,
    ) -> RetrievedChunk:
        """根据 DocumentChunk、分数和排名构造统一检索结果。"""

        return RetrievedChunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            content_hash=chunk.content_hash,
            version_id=chunk.version_id,
            text=chunk.text,
            score=score,
            rank=rank,
            retriever=retriever,
            source_path=chunk.source_path,
            chunk_index=chunk.chunk_index,
            title=chunk.title,
            section=chunk.section,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            metadata=chunk.metadata,
        )
