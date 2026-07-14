"""索引子系统运行时配置。

这里的 Config 是功能类真正接收的配置，不直接代表外部配置文件结构。
外部 TOML 读取出的 Settings 会在 factory 层转换成这些 Config。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


EmbeddingProvider = str
VectorRepositoryType = Literal["memory", "local_json"]
DistanceMetric = Literal["cosine"]


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding 客户端运行时配置。"""

    provider: EmbeddingProvider = "mock"
    model: str = "mock-hash-embedding"
    dimension: int = 16
    batch_size: int = 32
    timeout_seconds: float = 30.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("embedding provider 不能为空")
        if not self.model.strip():
            raise ValueError("embedding model 不能为空")
        if self.dimension <= 0:
            raise ValueError("embedding dimension 必须大于 0")
        if self.batch_size <= 0:
            raise ValueError("embedding batch_size 必须大于 0")
        if self.timeout_seconds <= 0:
            raise ValueError("embedding timeout_seconds 必须大于 0")
        if self.max_retries < 0:
            raise ValueError("embedding max_retries 必须大于等于 0")


@dataclass(frozen=True)
class VectorRepositoryConfig:
    """向量持久化运行时配置。"""

    repository_type: VectorRepositoryType = "memory"
    index_dir: Path = Path("data/indexes")
    collection_name: str = "papers_baseline"
    distance_metric: DistanceMetric = "cosine"
    persist: bool = False

    def __post_init__(self) -> None:
        if not self.collection_name.strip():
            raise ValueError("vector repository collection_name 不能为空")

    @property
    def collection_dir(self) -> Path:
        """当前 collection 的持久化目录。"""

        return self.index_dir / self.collection_name

    @property
    def vector_collection_path(self) -> Path:
        """本地 JSON 向量集合文件路径。"""

        return self.collection_dir / "vector_collection.json"

    @property
    def chunk_collection_path(self) -> Path:
        """本地 JSON chunk 集合文件路径。"""

        return self.collection_dir / "chunk_collection.json"

    @property
    def document_collection_path(self) -> Path:
        """本地 JSON 文档集合文件路径。"""

        return self.collection_dir / "document_collection.json"

    @property
    def embedding_cache_path(self) -> Path:
        """当前 collection 的 embedding cache 文件路径。"""

        return self.collection_dir / "embedding_cache.json"


@dataclass(frozen=True)
class IndexBuilderConfig:
    """索引构建流程运行时配置。"""

    manifest_filename: str = "manifest.json"
    build_report_filename: str = "index_build_report.json"
    skip_existing: bool = True
    fail_on_empty_chunk: bool = True

    def __post_init__(self) -> None:
        if not self.manifest_filename.strip():
            raise ValueError("manifest_filename 不能为空")
        if not self.build_report_filename.strip():
            raise ValueError("build_report_filename 不能为空")
