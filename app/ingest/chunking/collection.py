"""Chunk 运行时集合。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from app.core.models import DocumentChunk


class ChunkCollection(Protocol):
    """DocumentChunk 的运行时集合协议。"""

    def add(self, chunk: DocumentChunk) -> None:
        """保存单个 chunk。"""

    def add_many(self, chunks: Iterable[DocumentChunk]) -> None:
        """批量保存 chunks。"""

    def get_by_id(self, chunk_id: str) -> DocumentChunk | None:
        """根据 chunk_id 读取 chunk。"""

    def get_by_ids(self, chunk_ids: Iterable[str]) -> list[DocumentChunk]:
        """按输入顺序批量读取 chunks。"""

    def iter_chunks(self) -> Iterable[DocumentChunk]:
        """遍历当前集合中的所有 chunks。"""

    def count(self) -> int:
        """返回 chunk 数量。"""


class InMemoryChunkCollection:
    """内存 chunk 集合。"""

    def __init__(self) -> None:
        self._chunks: dict[str, DocumentChunk] = {}

    def add(self, chunk: DocumentChunk) -> None:
        """保存单个 chunk。"""

        self._chunks[chunk.chunk_id] = chunk

    def add_many(self, chunks: Iterable[DocumentChunk]) -> None:
        """批量保存 chunks。"""

        for chunk in chunks:
            self.add(chunk)

    def get_by_id(self, chunk_id: str) -> DocumentChunk | None:
        """根据 chunk_id 读取 chunk。"""

        return self._chunks.get(chunk_id)

    def get_by_ids(self, chunk_ids: Iterable[str]) -> list[DocumentChunk]:
        """按输入顺序批量读取 chunks。"""

        return [
            chunk
            for chunk_id in chunk_ids
            if (chunk := self.get_by_id(chunk_id)) is not None
        ]

    def iter_chunks(self) -> Iterable[DocumentChunk]:
        """遍历当前集合中的所有 chunks。"""

        return self._chunks.values()

    def count(self) -> int:
        """返回 chunk 数量。"""

        return len(self._chunks)
