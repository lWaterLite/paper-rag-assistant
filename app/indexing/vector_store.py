"""内存向量存储。

这是子模块 1 的练习版实现，用于理解 storing 和 retrieval 的基本形状。
真实项目后续可以替换为 FAISS、Chroma、Qdrant 或 pgvector。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.core.models import DocumentChunk, RetrievedChunk


@dataclass(frozen=True)
class VectorRecord:
    """向量库中的一条记录。"""

    chunk: DocumentChunk
    vector: list[float]


class InMemoryVectorStore:
    """简单内存向量库。"""

    def __init__(self) -> None:
        self._records: list[VectorRecord] = []

    def add(self, chunk: DocumentChunk, vector: list[float]) -> None:
        self._records.append(VectorRecord(chunk=chunk, vector=vector))

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        """基于余弦相似度搜索 top-k。"""

        scored = [
            (self._cosine_similarity(query_vector, record.vector), record.chunk)
            for record in self._records
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        results: list[RetrievedChunk] = []
        for rank, (score, chunk) in enumerate(scored[:top_k], start=1):
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    content_hash=chunk.content_hash,
                    version_id=chunk.version_id,
                    text=chunk.text,
                    score=round(score, 4),
                    rank=rank,
                    retriever="vector",
                    source_path=chunk.source_path,
                    title=chunk.title,
                    section=chunk.section,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    metadata=chunk.metadata,
                )
            )
        return results

    def count(self) -> int:
        return len(self._records)

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        """计算余弦相似度。"""

        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
        right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
        return dot / (left_norm * right_norm)

    # TODO 练习 6：
    # 当前 search 没有检查 query_vector 和记录向量的维度是否一致。
    # 请你补充维度检查，并为维度不一致设计一个清晰错误。
