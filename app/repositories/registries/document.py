"""文档 Repository 注册表。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from app.repositories.document import DocumentRepository
from app.repositories.registries._base import RepositoryRegistryBase

DocumentRepositoryBuilder = Callable[[Path], DocumentRepository]


class DocumentRepositoryRegistry(RepositoryRegistryBase[DocumentRepositoryBuilder]):
    """按持久化类型创建文档 Repository。"""

    def __init__(self) -> None:
        super().__init__(subject="document repository")

    def create(self, repository_type: str, *, path: Path) -> DocumentRepository:
        """根据类型和产物路径创建文档 Repository。"""

        return self.resolve(repository_type)(path)


def build_default_document_repository_registry() -> DocumentRepositoryRegistry:
    """创建项目内置的文档 Repository 注册表。"""

    from app.repositories.document import LocalJsonDocumentRepository

    registry = DocumentRepositoryRegistry()
    registry.register("local_json", LocalJsonDocumentRepository)
    return registry
