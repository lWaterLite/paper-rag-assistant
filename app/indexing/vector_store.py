"""向量存储。

当前提供两种实现：
1. InMemoryVectorStore：用于快速测试和进程内实验。
2. LocalJsonVectorStore：用于无第三方依赖的本地持久化 baseline。

后续接入 FAISS、Chroma、Qdrant 或 pgvector 时，应实现同一个 VectorStore 协议。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from app.core.errors import AppError, ErrorCode
from app.core.models import DocumentChunk, RetrievedChunk


@dataclass(frozen=True)
class VectorRecord:
    """向量库中的一条记录。"""

    chunk: DocumentChunk
    vector: list[float]


class VectorStore(Protocol):
    """向量存储协议。"""

    def add(self, chunk: DocumentChunk, vector: list[float]) -> None:
        """写入一个 chunk 向量。"""

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        """搜索最相似的 top-k chunk。"""

    def count(self) -> int:
        """返回向量数量。"""

    def contains_chunk(self, chunk_id: str) -> bool:
        """判断向量库中是否已经存在某个 chunk。"""

    @property
    def dimension(self) -> int | None:
        """返回向量库维度。"""

    def persist(self) -> None:
        """持久化向量库。内存实现可以是空操作。"""

    def load(self) -> None:
        """加载已有向量库。内存实现可以是空操作。"""


class BaseExactVectorStore:
    """基于精确余弦相似度的向量存储基类。"""

    def __init__(self) -> None:
        self._records_by_chunk_id: dict[str, VectorRecord] = {}
        self._dimension: int | None = None

    def add(self, chunk: DocumentChunk, vector: list[float]) -> None:
        """写入一个 chunk 向量。"""

        self._ensure_non_empty_vector(vector, ErrorCode.INDEX_FAILED, "写入向量")
        if self._dimension is None:
            self._dimension = len(vector)
        else:
            self._ensure_dimension(vector, ErrorCode.INDEX_FAILED, "写入向量")

        if chunk.chunk_id in self._records_by_chunk_id:
            return

        self._records_by_chunk_id[chunk.chunk_id] = VectorRecord(chunk=chunk, vector=vector)

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        """基于余弦相似度搜索 top-k。"""

        if not self._records_by_chunk_id or top_k <= 0:
            return []

        self._ensure_non_empty_vector(query_vector, ErrorCode.RETRIEVAL_FAILED, "查询向量")
        self._ensure_dimension(query_vector, ErrorCode.RETRIEVAL_FAILED, "查询向量")

        scored = [
            (self._cosine_similarity(query_vector, record.vector), record.chunk)
            for record in self._records_by_chunk_id.values()
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
                    chunk_index=chunk.chunk_index,
                    title=chunk.title,
                    section=chunk.section,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    metadata=chunk.metadata,
                )
            )
        return results

    def count(self) -> int:
        return len(self._records_by_chunk_id)

    def contains_chunk(self, chunk_id: str) -> bool:
        """判断向量库中是否已经存在某个 chunk。"""

        return chunk_id in self._records_by_chunk_id

    @property
    def dimension(self) -> int | None:
        """当前向量库维度。

        空向量库还没有维度，因此返回 None。
        """

        return self._dimension

    def persist(self) -> None:
        """默认不持久化。"""

    def load(self) -> None:
        """默认不加载外部状态。"""

    def _set_records(self, records: list[VectorRecord]) -> None:
        """加载持久化记录时重建内部索引。"""

        self._records_by_chunk_id = {}
        self._dimension = None
        for record in records:
            self.add(record.chunk, record.vector)

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


class InMemoryVectorStore(BaseExactVectorStore):
    """简单内存向量库。"""


class LocalJsonVectorStore(BaseExactVectorStore):
    """本地 JSON 向量库。

    这个实现使用精确余弦相似度，适合学习、测试和小规模论文集合。
    大规模生产场景应替换为专业向量库，但可以复用同一个 VectorStore 协议。
    """

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path
        self.load()

    def persist(self) -> None:
        """把向量记录写入 JSON 文件。

        目录创建由 IndexBuilder 在流程准备阶段完成。
        """

        payload = {
            "dimension": self._dimension,
            "records": [
                {
                    "chunk": asdict(record.chunk),
                    "vector": record.vector,
                }
                for record in sorted(
                    self._records_by_chunk_id.values(),
                    key=lambda item: item.chunk.chunk_id,
                )
            ],
        }
        self._path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> None:
        """从 JSON 文件加载已有向量记录。"""

        if not self._path.exists():
            return

        payload = json.loads(self._path.read_text(encoding="utf-8"))
        records = [
            VectorRecord(
                chunk=DocumentChunk(**item["chunk"]),
                vector=[float(value) for value in item["vector"]],
            )
            for item in payload.get("records", [])
        ]
        self._set_records(records)
