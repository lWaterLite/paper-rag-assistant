"""Indexing Settings 到运行时 Config 的适配。"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.settings import IndexingSettings
from app.indexing.configuration import (
    EmbeddingConfig,
    IndexBuilderConfig,
    VectorRepositoryConfig,
)


@dataclass(frozen=True, slots=True)
class IndexingConfigAdapter:
    """将 IndexingSettings 一次转换为可复用的运行时 Config 快照。"""

    settings: IndexingSettings
    embedding: EmbeddingConfig = field(init=False)
    vector_repository: VectorRepositoryConfig = field(init=False)
    index_builder: IndexBuilderConfig = field(init=False)

    def __post_init__(self) -> None:
        embedding_settings = self.settings.embedding
        object.__setattr__(
            self,
            "embedding",
            EmbeddingConfig(
                provider=embedding_settings.provider,
                model=embedding_settings.model,
                dimension=embedding_settings.dimension,
                batch_size=embedding_settings.batch_size,
                timeout_seconds=embedding_settings.timeout_seconds,
                max_retries=embedding_settings.max_retries,
            ),
        )
        repository_settings = self.settings.vector_repository
        object.__setattr__(
            self,
            "vector_repository",
            VectorRepositoryConfig(
                repository_type=repository_settings.type,
                index_dir=repository_settings.index_dir,
                collection_name=repository_settings.collection_name,
                distance_metric=repository_settings.distance_metric,
            ),
        )
        builder_settings = self.settings.builder
        object.__setattr__(
            self,
            "index_builder",
            IndexBuilderConfig(
                manifest_filename=builder_settings.manifest_filename,
                build_report_filename=builder_settings.build_report_filename,
                skip_existing=builder_settings.skip_existing,
                fail_on_empty_chunk=builder_settings.fail_on_empty_chunk,
            ),
        )
