"""索引版本 Manifest 领域模型。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, cast, get_args

CURRENT_INDEX_SCHEMA_VERSION = 4
CURRENT_VECTOR_COLLECTION_SCHEMA_VERSION = 1
APPLICATION_PACKAGE_NAME = "paper-rag-assistant"
IndexVersionStatus = Literal["building", "ready", "failed", "deprecated"]
BUILDING_INDEX_STATUS: IndexVersionStatus = "building"
READY_INDEX_STATUS: IndexVersionStatus = "ready"
FAILED_INDEX_STATUS: IndexVersionStatus = "failed"
VALID_INDEX_VERSION_STATUSES = frozenset(get_args(IndexVersionStatus))


@dataclass(frozen=True, slots=True)
class EmbeddingRuntimeCompatibility:
    """查询向量与已持久化向量可否比较的运行时条件。"""

    provider: str
    model: str
    dimension: int


@dataclass(frozen=True, slots=True)
class VectorCollectionRuntimeCompatibility:
    """向量集合的存储格式与检索语义条件。"""

    repository_type: str
    distance_metric: str
    schema_version: int


@dataclass(frozen=True, slots=True)
class IndexRuntimeCompatibility:
    """加载既有索引时必须匹配的运行时条件。"""

    embedding: EmbeddingRuntimeCompatibility
    vector_collection: VectorCollectionRuntimeCompatibility


@dataclass(frozen=True, slots=True)
class IndexArtifactDefinition:
    """定义索引产物内容与身份的输入条件。"""

    source_dir: str
    chunker: str
    chunk_size: int
    chunk_overlap: int
    runtime_compatibility: IndexRuntimeCompatibility


@dataclass(frozen=True, slots=True)
class IndexBuildProvenance:
    """记录构建执行方式，供审计和排障使用。"""

    built_at: str
    application_package: str
    application_version: str
    embedding_batch_size: int
    embedding_timeout_seconds: float
    embedding_max_retries: int


@dataclass(frozen=True, slots=True)
class IndexStorageLocator:
    """记录索引产物写入的位置，不参与兼容性或版本身份判断。"""

    collection_name: str


@dataclass(frozen=True, slots=True)
class IndexManifest:
    """一次索引构建的版本清单。

    Manifest 把运行兼容性、产物定义和构建溯源分开保存：只有
    ``runtime_compatibility`` 决定索引能否被当前运行时加载。
    """

    index_id: str
    schema_version: int
    status: IndexVersionStatus
    parent_index_id: str | None
    artifact_definition: IndexArtifactDefinition
    build_provenance: IndexBuildProvenance
    storage_locator: IndexStorageLocator
    document_count: int
    chunk_count: int
    vector_count: int
    artifact_definition_hash: str
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
        embedding_timeout_seconds: float = 30.0,
        embedding_max_retries: int = 2,
        vector_repository_type: str = "local_json",
        vector_collection_name: str = "default",
        distance_metric: str = "cosine",
        vector_collection_schema_version: int = CURRENT_VECTOR_COLLECTION_SCHEMA_VERSION,
        application_package: str = APPLICATION_PACKAGE_NAME,
        application_version: str | None = None,
        parent_index_id: str | None = None,
        status: IndexVersionStatus = READY_INDEX_STATUS,
    ) -> IndexManifest:
        """根据当前构建上下文创建 Manifest。"""

        normalized_status = _validate_status(status)
        artifact_definition = IndexArtifactDefinition(
            source_dir=_normalize_source_dir(source_dir),
            chunker=chunker,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            runtime_compatibility=IndexRuntimeCompatibility(
                embedding=EmbeddingRuntimeCompatibility(
                    provider=embedding_provider,
                    model=embedding_model,
                    dimension=embedding_dimension,
                ),
                vector_collection=VectorCollectionRuntimeCompatibility(
                    repository_type=vector_repository_type,
                    distance_metric=distance_metric,
                    schema_version=vector_collection_schema_version,
                ),
            ),
        )
        build_provenance = IndexBuildProvenance(
            built_at=datetime.now(UTC).isoformat(),
            application_package=application_package,
            application_version=(
                application_version
                if application_version is not None
                else _resolve_application_version(application_package)
            ),
            embedding_batch_size=embedding_batch_size,
            embedding_timeout_seconds=embedding_timeout_seconds,
            embedding_max_retries=embedding_max_retries,
        )
        storage_locator = IndexStorageLocator(collection_name=vector_collection_name)
        normalized_document_versions = dict(sorted(document_versions.items()))
        artifact_definition_hash = _hash_payload(asdict(artifact_definition))
        document_set_hash = _build_document_set_hash(normalized_document_versions)
        return cls(
            index_id=_build_index_id(
                artifact_definition_hash=artifact_definition_hash,
                document_set_hash=document_set_hash,
            ),
            schema_version=CURRENT_INDEX_SCHEMA_VERSION,
            status=normalized_status,
            parent_index_id=parent_index_id,
            artifact_definition=artifact_definition,
            build_provenance=build_provenance,
            storage_locator=storage_locator,
            document_count=document_count,
            chunk_count=chunk_count,
            vector_count=vector_count,
            artifact_definition_hash=artifact_definition_hash,
            document_set_hash=document_set_hash,
            document_versions=normalized_document_versions,
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IndexManifest:
        """从当前 Schema 的持久化字典恢复 Manifest。"""

        schema_version = int(_required(data, "schema_version"))
        if schema_version != CURRENT_INDEX_SCHEMA_VERSION:
            raise ValueError(
                "不支持的索引 Manifest Schema 版本："
                f"manifest={schema_version}，current={CURRENT_INDEX_SCHEMA_VERSION}；"
                "请重新构建索引"
            )

        artifact_definition_data = _required_mapping(data, "artifact_definition")
        runtime_data = _required_mapping(
            artifact_definition_data, "runtime_compatibility"
        )
        embedding_data = _required_mapping(runtime_data, "embedding")
        vector_collection_data = _required_mapping(runtime_data, "vector_collection")
        provenance_data = _required_mapping(data, "build_provenance")
        storage_data = _required_mapping(data, "storage_locator")
        return cls(
            index_id=str(_required(data, "index_id")),
            schema_version=schema_version,
            status=_validate_status(str(_required(data, "status"))),
            parent_index_id=_optional_string(data.get("parent_index_id")),
            artifact_definition=IndexArtifactDefinition(
                source_dir=str(_required(artifact_definition_data, "source_dir")),
                chunker=str(_required(artifact_definition_data, "chunker")),
                chunk_size=int(_required(artifact_definition_data, "chunk_size")),
                chunk_overlap=int(_required(artifact_definition_data, "chunk_overlap")),
                runtime_compatibility=IndexRuntimeCompatibility(
                    embedding=EmbeddingRuntimeCompatibility(
                        provider=str(_required(embedding_data, "provider")),
                        model=str(_required(embedding_data, "model")),
                        dimension=int(_required(embedding_data, "dimension")),
                    ),
                    vector_collection=VectorCollectionRuntimeCompatibility(
                        repository_type=str(
                            _required(vector_collection_data, "repository_type")
                        ),
                        distance_metric=str(
                            _required(vector_collection_data, "distance_metric")
                        ),
                        schema_version=int(
                            _required(vector_collection_data, "schema_version")
                        ),
                    ),
                ),
            ),
            build_provenance=IndexBuildProvenance(
                built_at=str(_required(provenance_data, "built_at")),
                application_package=str(
                    _required(provenance_data, "application_package")
                ),
                application_version=str(
                    _required(provenance_data, "application_version")
                ),
                embedding_batch_size=int(
                    _required(provenance_data, "embedding_batch_size")
                ),
                embedding_timeout_seconds=float(
                    _required(provenance_data, "embedding_timeout_seconds")
                ),
                embedding_max_retries=int(
                    _required(provenance_data, "embedding_max_retries")
                ),
            ),
            storage_locator=IndexStorageLocator(
                collection_name=str(_required(storage_data, "collection_name"))
            ),
            document_count=int(_required(data, "document_count")),
            chunk_count=int(_required(data, "chunk_count")),
            vector_count=int(_required(data, "vector_count")),
            artifact_definition_hash=str(_required(data, "artifact_definition_hash")),
            document_set_hash=str(_required(data, "document_set_hash")),
            document_versions={
                str(key): str(value)
                for key, value in _required_mapping(data, "document_versions").items()
            },
        )


def _build_index_id(*, artifact_definition_hash: str, document_set_hash: str) -> str:
    payload = "|".join(
        [str(CURRENT_INDEX_SCHEMA_VERSION), artifact_definition_hash, document_set_hash]
    )
    return f"idx_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"


def _build_document_set_hash(document_versions: dict[str, str]) -> str:
    return _hash_payload({"document_versions": dict(sorted(document_versions.items()))})


def _normalize_source_dir(source_dir: Path) -> str:
    return source_dir.expanduser().resolve(strict=False).as_posix()


def _resolve_application_version(application_package: str) -> str:
    """读取已安装应用包版本；源码直跑时明确记录未安装状态。"""

    try:
        return version(application_package)
    except PackageNotFoundError:
        return "uninstalled"


def _hash_payload(payload: object) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _required(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"索引 Manifest 缺少必需字段：{key}")
    return data[key]


def _required_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = _required(data, key)
    if not isinstance(value, dict):
        raise TypeError(f"索引 Manifest 字段必须是对象：{key}")
    return value


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
