"""索引 manifest 测试。"""

from __future__ import annotations

import unittest
import shutil
import uuid
from dataclasses import replace
from pathlib import Path

from app.indexing.configs import EmbeddingConfig, IndexBuilderConfig, VectorRepositoryConfig
from app.indexing.manifest import CURRENT_INDEX_SCHEMA_VERSION, IndexManifest, validate_manifest_compatible
from app.repositories.index_manifest_repository import IndexManifestRepository


class IndexManifestTest(unittest.TestCase):
    """验证 manifest 记录索引构建关键信息。"""

    def test_build_manifest_contains_reproducible_index_metadata(self) -> None:
        manifest = IndexManifest.build(
            source_dir=Path("data/raw/papers"),
            chunker="CharacterChunker",
            chunk_size=500,
            chunk_overlap=80,
            embedding_provider="mock",
            embedding_model="mock-hash-embedding",
            embedding_dimension=16,
            embedding_batch_size=8,
            vector_repository_type="local_json",
            vector_collection_name="papers",
            distance_metric="cosine",
            document_count=2,
            chunk_count=10,
            vector_count=10,
            document_versions={"doc_1": "v_1"},
        )

        self.assertTrue(manifest.index_id.startswith("idx_"))
        self.assertEqual(manifest.schema_version, CURRENT_INDEX_SCHEMA_VERSION)
        self.assertEqual(manifest.status, "ready")
        self.assertIsNone(manifest.parent_index_id)
        self.assertEqual(manifest.source_dir, Path("data/raw/papers").resolve(strict=False).as_posix())
        self.assertEqual(manifest.chunker, "CharacterChunker")
        self.assertEqual(manifest.embedding_provider, "mock")
        self.assertEqual(manifest.embedding_model, "mock-hash-embedding")
        self.assertEqual(manifest.embedding_dimension, 16)
        self.assertEqual(manifest.embedding_batch_size, 8)
        self.assertEqual(manifest.vector_repository_type, "local_json")
        self.assertEqual(manifest.vector_collection_name, "papers")
        self.assertEqual(manifest.distance_metric, "cosine")
        self.assertTrue(manifest.config_hash)
        self.assertTrue(manifest.document_set_hash)
        self.assertEqual(manifest.document_versions, {"doc_1": "v_1"})

    def test_manifest_can_be_converted_to_json_ready_dict(self) -> None:
        manifest = IndexManifest.build(
            source_dir=Path("data/raw/papers"),
            chunker="CharacterChunker",
            chunk_size=500,
            chunk_overlap=80,
            embedding_provider="mock",
            embedding_model="mock-hash-embedding",
            embedding_dimension=16,
            document_count=2,
            chunk_count=10,
            vector_count=10,
            document_versions={"doc_1": "v_1"},
        )

        data = manifest.to_dict()

        self.assertEqual(data["index_id"], manifest.index_id)
        self.assertEqual(data["schema_version"], CURRENT_INDEX_SCHEMA_VERSION)
        self.assertEqual(data["status"], "ready")
        self.assertIsNone(data["parent_index_id"])
        self.assertEqual(data["document_set_hash"], manifest.document_set_hash)
        self.assertEqual(data["document_versions"], {"doc_1": "v_1"})

    def test_relative_and_absolute_source_dir_share_same_index_version(self) -> None:
        relative_manifest = IndexManifest.build(
            source_dir=Path("data/raw/papers"),
            chunker="CharacterChunker",
            chunk_size=500,
            chunk_overlap=80,
            embedding_provider="mock",
            embedding_model="mock-hash-embedding",
            embedding_dimension=16,
            document_count=1,
            chunk_count=5,
            vector_count=5,
            document_versions={"doc_1": "v_1"},
        )
        absolute_manifest = IndexManifest.build(
            source_dir=Path("data/raw/papers").resolve(strict=False),
            chunker="CharacterChunker",
            chunk_size=500,
            chunk_overlap=80,
            embedding_provider="mock",
            embedding_model="mock-hash-embedding",
            embedding_dimension=16,
            document_count=1,
            chunk_count=5,
            vector_count=5,
            document_versions={"doc_1": "v_1"},
        )

        self.assertEqual(absolute_manifest.source_dir, relative_manifest.source_dir)
        self.assertEqual(absolute_manifest.config_hash, relative_manifest.config_hash)
        self.assertEqual(absolute_manifest.index_id, relative_manifest.index_id)

    def test_document_change_creates_new_index_version_without_changing_config_hash(self) -> None:
        base_manifest = IndexManifest.build(
            source_dir=Path("data/raw/papers"),
            chunker="CharacterChunker",
            chunk_size=500,
            chunk_overlap=80,
            embedding_provider="mock",
            embedding_model="mock-hash-embedding",
            embedding_dimension=16,
            document_count=1,
            chunk_count=5,
            vector_count=5,
            document_versions={"doc_1": "v_1"},
        )
        changed_manifest = IndexManifest.build(
            source_dir=Path("data/raw/papers"),
            chunker="CharacterChunker",
            chunk_size=500,
            chunk_overlap=80,
            embedding_provider="mock",
            embedding_model="mock-hash-embedding",
            embedding_dimension=16,
            document_count=1,
            chunk_count=5,
            vector_count=5,
            document_versions={"doc_1": "v_2"},
        )

        self.assertEqual(changed_manifest.config_hash, base_manifest.config_hash)
        self.assertNotEqual(changed_manifest.document_set_hash, base_manifest.document_set_hash)
        self.assertNotEqual(changed_manifest.index_id, base_manifest.index_id)

    def test_config_change_creates_new_index_version_without_changing_document_set_hash(self) -> None:
        base_manifest = IndexManifest.build(
            source_dir=Path("data/raw/papers"),
            chunker="CharacterChunker",
            chunk_size=500,
            chunk_overlap=80,
            embedding_provider="mock",
            embedding_model="mock-hash-embedding",
            embedding_dimension=16,
            document_count=1,
            chunk_count=5,
            vector_count=5,
            document_versions={"doc_1": "v_1"},
        )
        changed_manifest = IndexManifest.build(
            source_dir=Path("data/raw/papers"),
            chunker="CharacterChunker",
            chunk_size=700,
            chunk_overlap=80,
            embedding_provider="mock",
            embedding_model="mock-hash-embedding",
            embedding_dimension=16,
            document_count=1,
            chunk_count=5,
            vector_count=5,
            document_versions={"doc_1": "v_1"},
        )

        self.assertNotEqual(changed_manifest.config_hash, base_manifest.config_hash)
        self.assertEqual(changed_manifest.document_set_hash, base_manifest.document_set_hash)
        self.assertNotEqual(changed_manifest.index_id, base_manifest.index_id)

    def test_manifest_can_record_parent_index_version(self) -> None:
        manifest = IndexManifest.build(
            source_dir=Path("data/raw/papers"),
            chunker="CharacterChunker",
            chunk_size=500,
            chunk_overlap=80,
            embedding_provider="mock",
            embedding_model="mock-hash-embedding",
            embedding_dimension=16,
            document_count=1,
            chunk_count=5,
            vector_count=5,
            document_versions={"doc_1": "v_1"},
            parent_index_id="idx_parent",
        )

        self.assertEqual(manifest.parent_index_id, "idx_parent")

    def test_manifest_repository_writes_and_reads_manifest(self) -> None:
        index_dir = Path(".tmp_tests") / f"manifest_{uuid.uuid4().hex}"
        index_dir.mkdir(parents=True, exist_ok=True)
        try:
            repository = IndexManifestRepository(index_dir, IndexBuilderConfig(manifest_filename="manifest.json"))
            manifest = IndexManifest.build(
                source_dir=Path("data/raw/papers"),
                chunker="CharacterChunker",
                chunk_size=500,
                chunk_overlap=80,
                embedding_provider="mock",
                embedding_model="mock-hash-embedding",
                embedding_dimension=16,
                document_count=2,
                chunk_count=10,
                vector_count=10,
                document_versions={"doc_1": "v_1"},
            )

            path = repository.write(manifest)
            loaded = repository.read()

            self.assertEqual(path, index_dir / "manifest.json")
            self.assertEqual(loaded.index_id, manifest.index_id)
            self.assertEqual(loaded.config_hash, manifest.config_hash)
            self.assertEqual(loaded.document_set_hash, manifest.document_set_hash)
        finally:
            shutil.rmtree(index_dir, ignore_errors=True)

    def test_manifest_compatibility_rejects_dimension_mismatch(self) -> None:
        manifest = IndexManifest.build(
            source_dir=Path("data/raw/papers"),
            chunker="CharacterChunker",
            chunk_size=500,
            chunk_overlap=80,
            embedding_provider="mock",
            embedding_model="mock-hash-embedding",
            embedding_dimension=16,
            vector_repository_type="local_json",
            distance_metric="cosine",
            document_count=2,
            chunk_count=10,
            vector_count=10,
            document_versions={"doc_1": "v_1"},
        )

        with self.assertRaises(ValueError) as context:
            validate_manifest_compatible(
                manifest=manifest,
                embedding_config=EmbeddingConfig(provider="mock", model="mock-hash-embedding", dimension=32),
                vector_repository_config=VectorRepositoryConfig(repository_type="local_json", distance_metric="cosine"),
            )

        self.assertIn("embedding_dimension", str(context.exception))

    def test_manifest_compatibility_rejects_schema_version_mismatch(self) -> None:
        manifest = IndexManifest.build(
            source_dir=Path("data/raw/papers"),
            chunker="CharacterChunker",
            chunk_size=500,
            chunk_overlap=80,
            embedding_provider="mock",
            embedding_model="mock-hash-embedding",
            embedding_dimension=16,
            document_count=2,
            chunk_count=10,
            vector_count=10,
            document_versions={"doc_1": "v_1"},
        )

        with self.assertRaises(ValueError) as context:
            validate_manifest_compatible(
                manifest=replace(manifest, schema_version=CURRENT_INDEX_SCHEMA_VERSION + 1),
                embedding_config=EmbeddingConfig(provider="mock", model="mock-hash-embedding", dimension=16),
                vector_repository_config=VectorRepositoryConfig(),
            )

        self.assertIn("schema_version", str(context.exception))

    def test_manifest_compatibility_rejects_non_ready_status(self) -> None:
        manifest = IndexManifest.build(
            source_dir=Path("data/raw/papers"),
            chunker="CharacterChunker",
            chunk_size=500,
            chunk_overlap=80,
            embedding_provider="mock",
            embedding_model="mock-hash-embedding",
            embedding_dimension=16,
            document_count=2,
            chunk_count=10,
            vector_count=10,
            document_versions={"doc_1": "v_1"},
        )

        with self.assertRaises(ValueError) as context:
            validate_manifest_compatible(
                manifest=replace(manifest, status="failed"),
                embedding_config=EmbeddingConfig(provider="mock", model="mock-hash-embedding", dimension=16),
                vector_repository_config=VectorRepositoryConfig(),
            )

        self.assertIn("status", str(context.exception))


if __name__ == "__main__":
    unittest.main()
