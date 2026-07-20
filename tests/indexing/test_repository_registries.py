"""索引持久化 Repository 注册表测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

from app.indexing.configuration import IndexBuilderConfig
from app.repositories.chunk_repository import LocalJsonChunkRepository
from app.repositories.document_repository import LocalJsonDocumentRepository
from app.repositories.index_manifest_repository import IndexManifestRepository
from app.repositories.registries import (
    ChunkRepositoryRegistry,
    VectorRepositoryRegistry,
    build_default_chunk_repository_registry,
    build_default_document_repository_registry,
    build_default_manifest_repository_registry,
    build_default_vector_repository_registry,
)
from app.repositories.vector_repository import LocalJsonVectorRepository


class RepositoryRegistriesTest(unittest.TestCase):
    """验证所有索引持久化组件均通过领域 Registry 构建。"""

    def test_default_registries_create_local_json_repositories(self) -> None:
        root = Path(".tmp_tests/repository_registries")

        vector_repository = build_default_vector_repository_registry().create(
            "local_json",
            path=root / "vector_collection.json",
        )
        document_repository = build_default_document_repository_registry().create(
            "local_json",
            path=root / "document_collection.json",
        )
        chunk_repository = build_default_chunk_repository_registry().create(
            "local_json",
            path=root / "chunk_collection.json",
        )
        manifest_repository = build_default_manifest_repository_registry().create(
            "local_json",
            index_dir=root,
            config=IndexBuilderConfig(),
        )

        self.assertIsInstance(vector_repository, LocalJsonVectorRepository)
        self.assertIsInstance(document_repository, LocalJsonDocumentRepository)
        self.assertIsInstance(chunk_repository, LocalJsonChunkRepository)
        self.assertIsInstance(manifest_repository, IndexManifestRepository)

    def test_registry_rejects_unregistered_repository_type(self) -> None:
        registry = VectorRepositoryRegistry()

        with self.assertRaises(ValueError) as context:
            registry.create("memory", path=Path(".tmp_tests/vector_collection.json"))

        self.assertIn("不支持", str(context.exception))

    def test_registry_rejects_duplicate_registration_by_default(self) -> None:
        registry = ChunkRepositoryRegistry()
        builder = build_default_chunk_repository_registry().resolve("local_json")
        registry.register("local_json", builder)

        with self.assertRaises(ValueError) as context:
            registry.register("local_json", builder)

        self.assertIn("已注册", str(context.exception))


if __name__ == "__main__":
    unittest.main()
