"""向量集合持久化 Repository。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from app.core.errors import AppError, ErrorCode
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
        declared_dimension = _parse_declared_dimension(payload.get("dimension"), path=self._path)
        for item in payload.get("records", []):
            chunk_id = str(item["chunk_id"])
            vector = [float(value) for value in item["vector"]]
            _validate_record_dimension(
                path=self._path,
                chunk_id=chunk_id,
                vector=vector,
                declared_dimension=declared_dimension,
            )
            collection.add(
                VectorRecord(
                    chunk_id=chunk_id,
                    vector=vector,
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


def _parse_declared_dimension(value: Any, *, path: Path) -> int | None:
    """解析 JSON 中声明的向量维度。"""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise AppError(
            ErrorCode.INDEX_FAILED,
            f"向量集合文件 dimension 字段非法：{path.as_posix()}，dimension 必须是正整数或 null",
        )
    if value <= 0:
        raise AppError(
            ErrorCode.INDEX_FAILED,
            f"向量集合文件 dimension 字段非法：{path.as_posix()}，dimension 必须大于 0",
        )
    return value


def _validate_record_dimension(
        *,
        path: Path,
        chunk_id: str,
        vector: list[float],
        declared_dimension: int | None,
) -> None:
    """校验记录实际向量维度必须匹配 JSON 声明维度。"""

    if declared_dimension is None:
        return
    if len(vector) != declared_dimension:
        raise AppError(
            ErrorCode.INDEX_FAILED,
            f"向量集合文件维度不一致：{path.as_posix()} 中 chunk_id={chunk_id} "
            f"为 {len(vector)} 维，但文件声明 dimension={declared_dimension}",
        )
