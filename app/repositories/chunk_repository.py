"""Chunk 集合持久化 Repository。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from app.core.models import DocumentChunk
from app.ingest.chunking.collection import ChunkCollection, InMemoryChunkCollection


class ChunkRepository(Protocol):
    """Chunk 集合持久化协议。"""

    def load(self) -> ChunkCollection:
        """加载 chunk 集合。"""

    def save(self, collection: ChunkCollection) -> None:
        """保存 chunk 集合。"""


class LocalJsonChunkRepository:
    """基于本地 JSON 文件的 chunk 集合持久化。"""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> ChunkCollection:
        """从 JSON 文件加载 chunk 集合。"""

        collection = InMemoryChunkCollection()
        if not self._path.exists():
            return collection

        payload = json.loads(self._path.read_text(encoding="utf-8"))
        collection.add_many(DocumentChunk(**item) for item in payload.get("chunks", []))
        return collection

    def save(self, collection: ChunkCollection) -> None:
        """把 chunk 集合保存为 JSON 文件。"""

        payload = {
            "chunks": [
                asdict(chunk)
                for chunk in sorted(
                    collection.iter_chunks(), key=lambda item: item.chunk_id
                )
            ]
        }
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
