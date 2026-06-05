"""索引 manifest 测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

from app.indexing.manifest import IndexManifest


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


if __name__ == "__main__":
    unittest.main()

