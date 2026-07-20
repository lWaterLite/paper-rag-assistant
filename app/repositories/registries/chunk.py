"""Chunk Repository 注册表。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from app.repositories.chunk_repository import ChunkRepository, LocalJsonChunkRepository
from app.repositories.registries._base import RepositoryRegistryBase

ChunkRepositoryBuilder = Callable[[Path], ChunkRepository]


class ChunkRepositoryRegistry(RepositoryRegistryBase[ChunkRepositoryBuilder]):
    """按持久化类型创建 Chunk Repository。"""

    def __init__(self) -> None:
        super().__init__(subject="chunk repository")

    def create(self, repository_type: str, *, path: Path) -> ChunkRepository:
        """根据类型和产物路径创建 Chunk Repository。"""

        return self.resolve(repository_type)(path)


def build_default_chunk_repository_registry() -> ChunkRepositoryRegistry:
    """创建项目内置的 Chunk Repository 注册表。"""

    registry = ChunkRepositoryRegistry()
    registry.register("local_json", LocalJsonChunkRepository)
    return registry
