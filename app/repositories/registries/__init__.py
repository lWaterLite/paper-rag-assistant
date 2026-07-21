"""索引持久化 Repository 的注册与构建入口。"""

from app.repositories.registries.chunk import (
    ChunkRepositoryRegistry,
)
from app.repositories.registries.document import (
    DocumentRepositoryRegistry,
)
from app.repositories.registries.manifest import (
    ManifestRepositoryRegistry,
)
from app.repositories.registries.vector import (
    VectorRepositoryRegistry,
)

__all__ = [
    "ChunkRepositoryRegistry",
    "DocumentRepositoryRegistry",
    "ManifestRepositoryRegistry",
    "VectorRepositoryRegistry",
]
