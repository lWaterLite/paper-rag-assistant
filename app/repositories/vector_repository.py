"""向量集合持久化 Repository。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from app.indexing.vector_collection import InMemoryVectorCollection, VectorCollection, VectorRecord


class VectorRepository(Protocol):
    """向量集合持久化协议。"""

    def load(self) -> VectorCollection:
        """加载向量集合。"""

    def save(self, collection: VectorCollection) -> None:
        """保存向量集合。"""


class LocalJsonVectorRepository:
    """基于本地 JSON 文件的向量集合持久化。"""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> VectorCollection:
        """从 JSON 文件加载向量集合。"""

        collection = InMemoryVectorCollection()
        if not self._path.exists():
            return collection

        payload = json.loads(self._path.read_text(encoding="utf-8"))
        for item in payload.get("records", []):
            collection.add(
                VectorRecord(
                    chunk_id=str(item["chunk_id"]),
                    vector=[float(value) for value in item["vector"]],
                    metadata=dict(item.get("metadata", {})),
                )
            )
        return collection

    def save(self, collection: VectorCollection) -> None:
        """把向量集合保存为 JSON 文件。"""

        payload = {
            "dimension": collection.dimension,
            "records": [
                asdict(record)
                for record in sorted(collection.iter_records(), key=lambda item: item.chunk_id)
            ],
        }
        self._path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
