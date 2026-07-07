"""索引 manifest。

manifest 用来记录一次索引构建的关键配置和统计信息。
它的目标是让索引可以被解释、复现和比较。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast, get_args

from app.indexing.configs import EmbeddingConfig, VectorRepositoryConfig


CURRENT_INDEX_SCHEMA_VERSION = 3
IndexVersionStatus = Literal["building", "ready", "failed", "deprecated"]
BUILDING_INDEX_STATUS: IndexVersionStatus = "building"
READY_INDEX_STATUS: IndexVersionStatus = "ready"
FAILED_INDEX_STATUS: IndexVersionStatus = "failed"
VALID_INDEX_VERSION_STATUSES = frozenset(get_args(IndexVersionStatus))


@dataclass(frozen=True)
class IndexManifest:
    """索引构建清单。"""

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
            vector_repository_type: str = "memory",
            vector_collection_name: str = "default",
            distance_metric: str = "cosine",
            parent_index_id: str | None = None,
            status: IndexVersionStatus = READY_INDEX_STATUS,
    ) -> "IndexManifest":
        """根据索引构建参数创建 manifest。"""

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
        index_id = _build_index_id(
            config_hash=config_hash,
            document_set_hash=document_set_hash,
        )
        return cls(
            index_id=index_id,
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
        """转换为可 JSON 序列化的 dict。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IndexManifest":
        """从 JSON dict 恢复 manifest。"""

        status = _validate_status(str(_required(data, "status")))
        return cls(
            index_id=str(_required(data, "index_id")),
            schema_version=int(_required(data, "schema_version")),
            status=status,
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


def validate_manifest_compatible(
    *,
    manifest: IndexManifest,
    embedding_config: EmbeddingConfig,
    vector_repository_config: VectorRepositoryConfig,
) -> None:
    """校验已有索引 manifest 是否与当前核心配置兼容。"""

    mismatches: list[str] = []
    if manifest.schema_version != CURRENT_INDEX_SCHEMA_VERSION:
        mismatches.append(
            f"schema_version: manifest={manifest.schema_version}, current={CURRENT_INDEX_SCHEMA_VERSION}"
        )
    if manifest.status != READY_INDEX_STATUS:
        mismatches.append(f"status: manifest={manifest.status}, required={READY_INDEX_STATUS}")
    if manifest.embedding_provider != embedding_config.provider:
        mismatches.append(f"embedding_provider: manifest={manifest.embedding_provider}, current={embedding_config.provider}")
    if manifest.embedding_model != embedding_config.model:
        mismatches.append(f"embedding_model: manifest={manifest.embedding_model}, current={embedding_config.model}")
    if manifest.embedding_dimension != embedding_config.dimension:
        mismatches.append(f"embedding_dimension: manifest={manifest.embedding_dimension}, current={embedding_config.dimension}")
    if manifest.embedding_batch_size != embedding_config.batch_size:
        mismatches.append(
            f"embedding_batch_size: manifest={manifest.embedding_batch_size}, current={embedding_config.batch_size}"
        )
    if manifest.vector_repository_type != vector_repository_config.repository_type:
        mismatches.append(
            f"vector_repository_type: manifest={manifest.vector_repository_type}, "
            f"current={vector_repository_config.repository_type}"
        )
    if manifest.vector_collection_name != vector_repository_config.collection_name:
        mismatches.append(
            f"vector_collection_name: manifest={manifest.vector_collection_name}, "
            f"current={vector_repository_config.collection_name}"
        )
    if manifest.distance_metric != vector_repository_config.distance_metric:
        mismatches.append(f"distance_metric: manifest={manifest.distance_metric}, current={vector_repository_config.distance_metric}")
    if mismatches:
        raise ValueError("索引 manifest 与当前配置不兼容：" + "；".join(mismatches))


def _build_index_id(
        *,
        config_hash: str,
        document_set_hash: str,
) -> str:
    """生成稳定 index_id。

    index_id 是一次索引版本的稳定身份。
    schema_version、config_hash 和 document_set_hash 任意一个变化，都会得到新 index_id。
    """

    payload = "|".join(
        [
            str(CURRENT_INDEX_SCHEMA_VERSION),
            config_hash,
            document_set_hash,
        ]
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"idx_{digest}"


def _build_document_set_hash(document_versions: dict[str, str]) -> str:
    """对输入文档集合生成稳定 hash。"""

    return _hash_payload({"document_versions": dict(sorted(document_versions.items()))})


def _normalize_source_dir(source_dir: Path) -> str:
    """把文档目录路径规范化为稳定的 POSIX 字符串。

    manifest 的 source_dir 会参与 config_hash。
    如果不先规范化，`data/raw/papers` 和它对应的绝对路径会生成不同索引版本。
    """

    return source_dir.expanduser().resolve(strict=False).as_posix()


def _hash_payload(payload: dict[str, Any]) -> str:
    """对配置 payload 生成稳定 hash。"""

    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _required(data: dict[str, Any], key: str) -> Any:
    """读取 manifest 必需字段，缺失时给出更清晰的错误。"""

    if key not in data:
        raise ValueError(f"索引 manifest 缺少必需字段：{key}")
    return data[key]


def _optional_string(value: Any) -> str | None:
    """把可选字符串字段规范化。"""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _validate_status(status: str) -> IndexVersionStatus:
    """校验索引版本状态。"""

    if status not in VALID_INDEX_VERSION_STATUSES:
        allowed = ", ".join(sorted(VALID_INDEX_VERSION_STATUSES))
        raise ValueError(f"非法索引版本状态：{status}，可选值：{allowed}")
    return cast(IndexVersionStatus, status)
