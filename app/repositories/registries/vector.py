"""向量 Repository 注册表。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from app.repositories.registries._base import RepositoryRegistryBase
from app.repositories.vector import VectorRepository

VectorRepositoryBuilder = Callable[[Path], VectorRepository]


class VectorRepositoryRegistry(RepositoryRegistryBase[VectorRepositoryBuilder]):
    """按持久化类型创建向量 Repository。"""

    def __init__(self) -> None:
        super().__init__(subject="vector repository")

    def create(self, repository_type: str, *, path: Path) -> VectorRepository:
        """根据类型和产物路径创建向量 Repository。"""

        return self.resolve(repository_type)(path)


def build_default_vector_repository_registry() -> VectorRepositoryRegistry:
    """创建项目内置的向量 Repository 注册表。"""

    from app.repositories.vector import LocalJsonVectorRepository

    registry = VectorRepositoryRegistry()
    registry.register("local_json", LocalJsonVectorRepository)
    return registry
