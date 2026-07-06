"""索引 manifest 测试。"""

from __future__ import annotations

import unittest
import shutil
import uuid
from pathlib import Path

from app.indexing.configs import EmbeddingConfig, IndexBuilderConfig, VectorRepositoryConfig
from app.indexing.manifest import IndexManifest, IndexManifestStore, validate_manifest_compatible


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
        self.assertEqual(manifest.source_dir, "data/raw/papers")
        self.assertEqual(manifest.chunker, "CharacterChunker")
        self.assertEqual(manifest.embedding_provider, "mock")
        self.assertEqual(manifest.embedding_model, "mock-hash-embedding")
        self.assertEqual(manifest.embedding_dimension, 16)
        self.assertEqual(manifest.embedding_batch_size, 8)
        self.assertEqual(manifest.vector_repository_type, "local_json")
        self.assertEqual(manifest.vector_collection_name, "papers")
        self.assertEqual(manifest.distance_metric, "cosine")
        self.assertTrue(manifest.config_hash)
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
        self.assertEqual(data["document_versions"], {"doc_1": "v_1"})

    def test_manifest_store_writes_and_reads_manifest(self) -> None:
        index_dir = Path(".tmp_tests") / f"manifest_{uuid.uuid4().hex}"
        index_dir.mkdir(parents=True, exist_ok=True)
        try:
            store = IndexManifestStore(index_dir, IndexBuilderConfig(manifest_filename="manifest.json"))
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

            path = store.write(manifest)
            loaded = store.read()

            self.assertEqual(path, index_dir / "manifest.json")
            self.assertEqual(loaded.index_id, manifest.index_id)
            self.assertEqual(loaded.config_hash, manifest.config_hash)
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


if __name__ == "__main__":
    unittest.main()
