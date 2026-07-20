"""索引版本 Manifest 领域模型。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast, get_args


CURRENT_INDEX_SCHEMA_VERSION = 3
IndexVersionStatus = Literal["building", "ready", "failed", "deprecated"]
BUILDING_INDEX_STATUS: IndexVersionStatus = "building"
READY_INDEX_STATUS: IndexVersionStatus = "ready"
FAILED_INDEX_STATUS: IndexVersionStatus = "failed"
VALID_INDEX_VERSION_STATUSES = frozenset(get_args(IndexVersionStatus))


@dataclass(frozen=True)
class IndexManifest:
    """一次索引构建的可复现版本清单。"""

    index_id: str
    schema_version: int
    status: IndexVersionStatus
    parent_index_id: str | None
    source_dir: str
    created_at: str
    chunker: str
    chunk_size: int
    chunk_overlap: int
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    embedding_batch_size: int
    vector_repository_type: str
    vector_collection_name: str
    distance_metric: str
    document_count: int
    chunk_count: int
    vector_count: int
    config_hash: str
    document_set_hash: str
    document_versions: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        source_dir: Path,
        chunker: str,
        chunk_size: int,
        chunk_overlap: int,
        embedding_provider: str,
        embedding_model: str,
        embedding_dimension: int,
        document_count: int,
        chunk_count: int,
        vector_count: int,
        document_versions: dict[str, str],
        embedding_batch_size: int = 1,
        vector_repository_type: str = "local_json",
        vector_collection_name: str = "default",
        distance_metric: str = "cosine",
        parent_index_id: str | None = None,
        status: IndexVersionStatus = READY_INDEX_STATUS,
    ) -> "IndexManifest":
        """根据当前构建参数创建索引 Manifest。"""

        normalized_status = _validate_status(status)
        normalized_source_dir = _normalize_source_dir(source_dir)
        normalized_document_versions = dict(sorted(document_versions.items()))
        config_payload = {
            "source_dir": normalized_source_dir,
            "chunker": chunker,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
            "embedding_batch_size": embedding_batch_size,
            "vector_repository_type": vector_repository_type,
            "vector_collection_name": vector_collection_name,
            "distance_metric": distance_metric,
        }
        config_hash = _hash_payload(config_payload)
        document_set_hash = _build_document_set_hash(normalized_document_versions)
        return cls(
            index_id=_build_index_id(
                config_hash=config_hash,
                document_set_hash=document_set_hash,
            ),
            schema_version=CURRENT_INDEX_SCHEMA_VERSION,
            status=normalized_status,
            parent_index_id=parent_index_id,
            source_dir=normalized_source_dir,
            created_at=datetime.now(UTC).isoformat(),
            chunker=chunker,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            embedding_batch_size=embedding_batch_size,
            vector_repository_type=vector_repository_type,
            vector_collection_name=vector_collection_name,
            distance_metric=distance_metric,
            document_count=document_count,
            chunk_count=chunk_count,
            vector_count=vector_count,
            config_hash=config_hash,
            document_set_hash=document_set_hash,
            document_versions=normalized_document_versions,
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IndexManifest":
        """从持久化字典恢复 Manifest。"""

        return cls(
            index_id=str(_required(data, "index_id")),
            schema_version=int(_required(data, "schema_version")),
            status=_validate_status(str(_required(data, "status"))),
            parent_index_id=_optional_string(data.get("parent_index_id")),
            source_dir=str(_required(data, "source_dir")),
            created_at=str(_required(data, "created_at")),
            chunker=str(_required(data, "chunker")),
            chunk_size=int(_required(data, "chunk_size")),
            chunk_overlap=int(_required(data, "chunk_overlap")),
            embedding_provider=str(_required(data, "embedding_provider")),
            embedding_model=str(_required(data, "embedding_model")),
            embedding_dimension=int(_required(data, "embedding_dimension")),
            embedding_batch_size=int(_required(data, "embedding_batch_size")),
            vector_repository_type=str(_required(data, "vector_repository_type")),
            vector_collection_name=str(_required(data, "vector_collection_name")),
            distance_metric=str(_required(data, "distance_metric")),
            document_count=int(_required(data, "document_count")),
            chunk_count=int(_required(data, "chunk_count")),
            vector_count=int(_required(data, "vector_count")),
            config_hash=str(_required(data, "config_hash")),
            document_set_hash=str(_required(data, "document_set_hash")),
            document_versions={
                str(key): str(value)
                for key, value in _required(data, "document_versions").items()
            },
        )


def _build_index_id(*, config_hash: str, document_set_hash: str) -> str:
    payload = "|".join(
        [str(CURRENT_INDEX_SCHEMA_VERSION), config_hash, document_set_hash]
    )
    return f"idx_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"


def _build_document_set_hash(document_versions: dict[str, str]) -> str:
    return _hash_payload({"document_versions": dict(sorted(document_versions.items()))})


def _normalize_source_dir(source_dir: Path) -> str:
    return source_dir.expanduser().resolve(strict=False).as_posix()


def _hash_payload(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _required(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"索引 manifest 缺少必需字段：{key}")
    return data[key]


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _validate_status(status: str) -> IndexVersionStatus:
    if status not in VALID_INDEX_VERSION_STATUSES:
        allowed = ", ".join(sorted(VALID_INDEX_VERSION_STATUSES))
        raise ValueError(f"非法索引版本状态：{status}，可选值：{allowed}")
    return cast(IndexVersionStatus, status)
