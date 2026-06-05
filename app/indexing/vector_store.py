"""内存向量存储。

这是子模块 1 的练习版实现，用于理解 storing 和 retrieval 的基本形状。
真实项目后续可以替换为 FAISS、Chroma、Qdrant 或 pgvector。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.core.errors import AppError, ErrorCode
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
        self._chunk_ids: set[str] = set()
        self._dimension: int | None = None

    def add(self, chunk: DocumentChunk, vector: list[float]) -> None:
        """写入一个 chunk 向量。"""

        self._ensure_non_empty_vector(vector, ErrorCode.INDEX_FAILED, "写入向量")
        if self._dimension is None:
            self._dimension = len(vector)
        else:
            self._ensure_dimension(vector, ErrorCode.INDEX_FAILED, "写入向量")

        if chunk.chunk_id in self._chunk_ids:
            return

        self._records.append(VectorRecord(chunk=chunk, vector=vector))
        self._chunk_ids.add(chunk.chunk_id)

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        """基于余弦相似度搜索 top-k。"""

        if not self._records:
            return []

        self._ensure_non_empty_vector(query_vector, ErrorCode.RETRIEVAL_FAILED, "查询向量")
        self._ensure_dimension(query_vector, ErrorCode.RETRIEVAL_FAILED, "查询向量")

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

    def contains_chunk(self, chunk_id: str) -> bool:
        """判断向量库中是否已经存在某个 chunk。"""

        return chunk_id in self._chunk_ids

    @property
    def dimension(self) -> int | None:
        """当前向量库维度。

        空向量库还没有维度，因此返回 None。
        """

        return self._dimension

    def _ensure_dimension(self, vector: list[float], code: ErrorCode, vector_name: str) -> None:
        """校验向量维度必须与向量库维度一致。"""

        if self._dimension is None:
            return
        if len(vector) != self._dimension:
            raise AppError(
                code=code,
                message=f"{vector_name}与向量库维度不一致：{vector_name}为 {len(vector)} 维，向量库为 {self._dimension} 维",
            )

    @staticmethod
    def _ensure_non_empty_vector(vector: list[float], code: ErrorCode, vector_name: str) -> None:
        """校验向量不能为空。"""

        if not vector:
            raise AppError(
                code=code,
                message=f"{vector_name}不能为空",
            )

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        """计算余弦相似度。"""

        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
        right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
        return dot / (left_norm * right_norm)
