"""索引 Manifest Repository 注册表。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from app.indexing.configuration import IndexBuilderConfig
from app.repositories.index_manifest_repository import (
    IndexManifestRepository,
    ManifestRepository,
)
from app.repositories.registries._base import RepositoryRegistryBase

ManifestRepositoryBuilder = Callable[[Path, IndexBuilderConfig], ManifestRepository]


class ManifestRepositoryRegistry(RepositoryRegistryBase[ManifestRepositoryBuilder]):
    """按持久化类型创建 Manifest Repository。"""

    def __init__(self) -> None:
        super().__init__(subject="manifest repository")

    def create(
        self,
        repository_type: str,
        *,
        index_dir: Path,
        config: IndexBuilderConfig,
    ) -> ManifestRepository:
        """根据类型、索引目录和构建配置创建 Manifest Repository。"""

        return self.resolve(repository_type)(index_dir, config)


def build_default_manifest_repository_registry() -> ManifestRepositoryRegistry:
    """创建项目内置的 Manifest Repository 注册表。"""

    registry = ManifestRepositoryRegistry()
    registry.register("local_json", IndexManifestRepository)
    return registry
