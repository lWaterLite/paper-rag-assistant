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
from typing import Any

from app.indexing.configs import EmbeddingConfig, IndexBuilderConfig, VectorRepositoryConfig


@dataclass(frozen=True)
class IndexManifest:
    """索引构建清单。"""

    index_id: str
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
    ) -> "IndexManifest":
        """根据索引构建参数创建 manifest。"""

        config_payload = {
            "source_dir": source_dir.as_posix(),
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
            "document_versions": dict(sorted(document_versions.items())),
        }
        config_hash = _hash_payload(config_payload)
        index_id = _build_index_id(
            source_dir=source_dir,
            chunker=chunker,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            config_hash=config_hash,
        )
        return cls(
            index_id=index_id,
            source_dir=source_dir.as_posix(),
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
            document_versions=document_versions,
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的 dict。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IndexManifest":
        """从 JSON dict 恢复 manifest。"""

        return cls(
            index_id=str(data["index_id"]),
            source_dir=str(data["source_dir"]),
            created_at=str(data["created_at"]),
            chunker=str(data["chunker"]),
            chunk_size=int(data["chunk_size"]),
            chunk_overlap=int(data["chunk_overlap"]),
            embedding_provider=str(data["embedding_provider"]),
            embedding_model=str(data["embedding_model"]),
            embedding_dimension=int(data["embedding_dimension"]),
            embedding_batch_size=int(data.get("embedding_batch_size", 1)),
            vector_repository_type=str(data["vector_repository_type"]),
            vector_collection_name=str(data.get("vector_collection_name", "default")),
            distance_metric=str(data.get("distance_metric", "cosine")),
            document_count=int(data["document_count"]),
            chunk_count=int(data["chunk_count"]),
            vector_count=int(data["vector_count"]),
            config_hash=str(data.get("config_hash", "")),
            document_versions={str(key): str(value) for key, value in data.get("document_versions", {}).items()},
        )


class IndexManifestStore:
    """索引 manifest 读写器。"""

    def __init__(self, index_dir: Path, config: IndexBuilderConfig) -> None:
        self._index_dir = index_dir
        self._config = config

    @property
    def manifest_path(self) -> Path:
        """manifest 文件路径。"""

        return self._index_dir / self._config.manifest_filename

    def write(self, manifest: IndexManifest) -> Path:
        """写入 manifest 并返回文件路径。"""

        self.manifest_path.write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.manifest_path

    def read(self) -> IndexManifest:
        """读取 manifest。"""

        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return IndexManifest.from_dict(data)

    def exists(self) -> bool:
        """判断 manifest 是否存在。"""

        return self.manifest_path.exists()


def validate_manifest_compatible(
    *,
    manifest: IndexManifest,
    embedding_config: EmbeddingConfig,
    vector_repository_config: VectorRepositoryConfig,
) -> None:
    """校验已有索引 manifest 是否与当前核心配置兼容。"""

    mismatches: list[str] = []
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
        source_dir: Path,
        chunker: str,
        chunk_size: int,
        chunk_overlap: int,
        embedding_provider: str,
        embedding_model: str,
        embedding_dimension: int,
        config_hash: str | None = None,
) -> str:
    """生成稳定 index_id。

    可读字段保留在 payload 中，config_hash 则让文档版本等细节变化也能影响 index_id。
    """

    payload = "|".join(
        [
            source_dir.as_posix(),
            chunker,
            str(chunk_size),
            str(chunk_overlap),
            embedding_provider,
            embedding_model,
            str(embedding_dimension),
            config_hash or "",
        ]
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"idx_{digest}"


def _hash_payload(payload: dict[str, Any]) -> str:
    """对配置 payload 生成稳定 hash。"""

    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
