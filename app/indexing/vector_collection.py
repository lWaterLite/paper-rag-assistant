"""向量运行时集合。

Collection 只管理已经加载到内存中的向量记录，负责相似度搜索，不负责文件或数据库读写。
持久化读写由 app.repositories.vector_repository 中的 Repository 负责。
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.core.errors import AppError, ErrorCode


@dataclass(frozen=True)
class VectorRecord:
    """内存向量集合中的一条记录。"""

    chunk_id: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorSearchResult:
    """向量检索命中的轻量结果。"""

    chunk_id: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorCollection(Protocol):
    """向量运行时集合协议。"""

    def add(self, record: VectorRecord) -> None:
        """写入一条向量记录。"""

    def search(self, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
        """搜索最相似的 top-k 向量。"""

    def count(self) -> int:
        """返回向量数量。"""

    def contains_chunk(self, chunk_id: str) -> bool:
        """判断集合中是否存在某个 chunk 的向量。"""

    def iter_records(self) -> Iterable[VectorRecord]:
        """遍历所有向量记录。"""

    @property
    def dimension(self) -> int | None:
        """返回向量维度。"""


class InMemoryVectorCollection:
    """基于内存和精确余弦相似度的向量集合。"""

    def __init__(self) -> None:
        self._records_by_chunk_id: dict[str, VectorRecord] = {}
        self._dimension: int | None = None

    def add(self, record: VectorRecord) -> None:
        """写入一条向量记录。"""

        self._ensure_non_empty_vector(record.vector, ErrorCode.INDEX_FAILED, "写入向量")
        if self._dimension is None:
            self._dimension = len(record.vector)
        else:
            self._ensure_dimension(record.vector, ErrorCode.INDEX_FAILED, "写入向量")

        if record.chunk_id in self._records_by_chunk_id:
            return

        self._records_by_chunk_id[record.chunk_id] = record

    def search(self, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
        """基于余弦相似度搜索 top-k。"""

        if not self._records_by_chunk_id or top_k <= 0:
            return []

        self._ensure_non_empty_vector(query_vector, ErrorCode.RETRIEVAL_FAILED, "查询向量")
        self._ensure_dimension(query_vector, ErrorCode.RETRIEVAL_FAILED, "查询向量")

        scored = [
            (self._cosine_similarity(query_vector, record.vector), record)
            for record in self._records_by_chunk_id.values()
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        return [
            VectorSearchResult(
                chunk_id=record.chunk_id,
                score=round(score, 4),
                rank=rank,
                metadata=record.metadata,
            )
            for rank, (score, record) in enumerate(scored[:top_k], start=1)
        ]

    def count(self) -> int:
        return len(self._records_by_chunk_id)

    def contains_chunk(self, chunk_id: str) -> bool:
        """判断集合中是否存在某个 chunk 的向量。"""

        return chunk_id in self._records_by_chunk_id

    def iter_records(self) -> Iterable[VectorRecord]:
        """遍历所有向量记录。"""

        return self._records_by_chunk_id.values()

    @property
    def dimension(self) -> int | None:
        """当前向量集合维度。"""

        return self._dimension

    def _ensure_dimension(self, vector: list[float], code: ErrorCode, vector_name: str) -> None:
        """校验向量维度必须与集合维度一致。"""

        if self._dimension is None:
            return
        if len(vector) != self._dimension:
            raise AppError(
                code=code,
                message=f"{vector_name}与向量集合维度不一致：{vector_name}为 {len(vector)} 维，集合为 {self._dimension} 维",
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
