"""索引 Manifest 测试。"""

from __future__ import annotations

import shutil
import unittest
import uuid
from dataclasses import replace
from pathlib import Path

from app.indexing.configuration import (
    EmbeddingConfig,
    IndexBuilderConfig,
    VectorRepositoryConfig,
)
from app.indexing.manifests import IndexManifest
from app.indexing.manifests.compatibility import validate_manifest_compatible
from app.indexing.manifests.models import (
    CURRENT_INDEX_SCHEMA_VERSION,
)
from app.repositories.manifest import IndexManifestRepository


class IndexManifestTest(unittest.TestCase):
    """验证 Manifest 的产物身份、运行兼容性和构建溯源边界。"""

    def test_build_manifest_separates_runtime_and_provenance(self) -> None:
        manifest = _build_manifest(
            embedding_batch_size=8,
            embedding_timeout_seconds=12.5,
            embedding_max_retries=4,
            application_version="0.1.0-test",
        )
        artifact_definition = manifest.artifact_definition
        runtime = artifact_definition.runtime_compatibility

        self.assertTrue(manifest.index_id.startswith("idx_"))
        self.assertEqual(manifest.schema_version, CURRENT_INDEX_SCHEMA_VERSION)
        self.assertEqual(artifact_definition.chunker, "CharacterChunker")
        self.assertEqual(
            artifact_definition.source_dir,
            Path("data/raw/papers").resolve(strict=False).as_posix(),
        )
        self.assertEqual(runtime.embedding.provider, "mock")
        self.assertEqual(runtime.embedding.model, "mock-hash-embedding")
        self.assertEqual(runtime.embedding.dimension, 16)
        self.assertEqual(runtime.vector_collection.repository_type, "local_json")
        self.assertEqual(runtime.vector_collection.distance_metric, "cosine")
        self.assertEqual(manifest.storage_locator.collection_name, "papers")
        self.assertEqual(manifest.build_provenance.embedding_batch_size, 8)
        self.assertEqual(manifest.build_provenance.embedding_timeout_seconds, 12.5)
        self.assertEqual(manifest.build_provenance.embedding_max_retries, 4)
        self.assertEqual(manifest.build_provenance.application_version, "0.1.0-test")
        self.assertTrue(manifest.artifact_definition_hash)

    def test_manifest_serialization_round_trips_current_schema(self) -> None:
        manifest = _build_manifest()

        data = manifest.to_dict()
        loaded = IndexManifest.from_dict(data)

        self.assertIn("artifact_definition", data)
        self.assertIn("build_provenance", data)
        self.assertIn("storage_locator", data)
        self.assertNotIn("config_hash", data)
        self.assertEqual(loaded, manifest)

    def test_manifest_rejects_non_mapping_field_with_type_error(self) -> None:
        data = _build_manifest().to_dict()
        data["artifact_definition"] = []

        with self.assertRaises(TypeError) as context:
            IndexManifest.from_dict(data)

        self.assertIn("artifact_definition", str(context.exception))

    def test_relative_and_absolute_source_dir_share_same_index_version(self) -> None:
        relative_manifest = _build_manifest(source_dir=Path("data/raw/papers"))
        absolute_manifest = _build_manifest(
            source_dir=Path("data/raw/papers").resolve(strict=False)
        )

        self.assertEqual(
            absolute_manifest.artifact_definition.source_dir,
            relative_manifest.artifact_definition.source_dir,
        )
        self.assertEqual(
            absolute_manifest.artifact_definition_hash,
            relative_manifest.artifact_definition_hash,
        )
        self.assertEqual(absolute_manifest.index_id, relative_manifest.index_id)

    def test_document_change_creates_new_version_without_changing_artifact_definition(
        self,
    ) -> None:
        base_manifest = _build_manifest(document_versions={"doc_1": "v_1"})
        changed_manifest = _build_manifest(document_versions={"doc_1": "v_2"})

        self.assertEqual(
            changed_manifest.artifact_definition_hash,
            base_manifest.artifact_definition_hash,
        )
        self.assertNotEqual(
            changed_manifest.document_set_hash, base_manifest.document_set_hash
        )
        self.assertNotEqual(changed_manifest.index_id, base_manifest.index_id)

    def test_chunking_change_creates_new_index_version(self) -> None:
        base_manifest = _build_manifest(chunk_size=500)
        changed_manifest = _build_manifest(chunk_size=700)

        self.assertNotEqual(
            changed_manifest.artifact_definition_hash,
            base_manifest.artifact_definition_hash,
        )
        self.assertEqual(
            changed_manifest.document_set_hash, base_manifest.document_set_hash
        )
        self.assertNotEqual(changed_manifest.index_id, base_manifest.index_id)

    def test_embedding_execution_change_preserves_index_identity(self) -> None:
        base_manifest = _build_manifest(
            embedding_batch_size=8,
            embedding_timeout_seconds=10,
            embedding_max_retries=1,
        )
        changed_manifest = _build_manifest(
            embedding_batch_size=64,
            embedding_timeout_seconds=60,
            embedding_max_retries=5,
        )

        self.assertNotEqual(
            changed_manifest.build_provenance, base_manifest.build_provenance
        )
        self.assertEqual(
            changed_manifest.artifact_definition_hash,
            base_manifest.artifact_definition_hash,
        )
        self.assertEqual(changed_manifest.index_id, base_manifest.index_id)

    def test_manifest_compatibility_allows_embedding_execution_change(self) -> None:
        manifest = _build_manifest(embedding_batch_size=8)

        validate_manifest_compatible(
            manifest=manifest,
            embedding_config=EmbeddingConfig(
                provider="mock",
                model="mock-hash-embedding",
                dimension=16,
                batch_size=64,
                timeout_seconds=90,
                max_retries=5,
            ),
            vector_repository_config=VectorRepositoryConfig(
                repository_type="local_json",
                distance_metric="cosine",
            ),
        )

    def test_manifest_compatibility_rejects_runtime_mismatch(self) -> None:
        manifest = _build_manifest()

        with self.assertRaises(ValueError) as context:
            validate_manifest_compatible(
                manifest=manifest,
                embedding_config=EmbeddingConfig(
                    provider="mock",
                    model="mock-hash-embedding",
                    dimension=32,
                ),
                vector_repository_config=VectorRepositoryConfig(
                    repository_type="local_json",
                    distance_metric="cosine",
                ),
            )

        self.assertIn("embedding_dimension", str(context.exception))

    def test_manifest_repository_writes_and_reads_manifest(self) -> None:
        index_dir = Path(".tmp_tests") / f"manifest_{uuid.uuid4().hex}"
        index_dir.mkdir(parents=True, exist_ok=True)
        try:
            repository = IndexManifestRepository(
                index_dir,
                IndexBuilderConfig(manifest_filename="manifest.json"),
            )
            manifest = _build_manifest()

            path = repository.write(manifest)
            loaded = repository.read()

            self.assertEqual(path, index_dir / "manifest.json")
            self.assertEqual(loaded.index_id, manifest.index_id)
            self.assertEqual(
                loaded.artifact_definition_hash,
                manifest.artifact_definition_hash,
            )
            self.assertEqual(loaded.document_set_hash, manifest.document_set_hash)
        finally:
            shutil.rmtree(index_dir, ignore_errors=True)

    def test_v3_manifest_is_rejected_with_rebuild_guidance(self) -> None:
        legacy_data = _build_manifest().to_dict()
        legacy_data["schema_version"] = CURRENT_INDEX_SCHEMA_VERSION - 1

        with self.assertRaises(ValueError) as context:
            IndexManifest.from_dict(legacy_data)

        self.assertIn("请重新构建索引", str(context.exception))

    def test_manifest_compatibility_rejects_non_ready_status(self) -> None:
        manifest = _build_manifest()

        with self.assertRaises(ValueError) as context:
            validate_manifest_compatible(
                manifest=replace(manifest, status="failed"),
                embedding_config=EmbeddingConfig(
                    provider="mock", model="mock-hash-embedding", dimension=16
                ),
                vector_repository_config=VectorRepositoryConfig(),
            )

        self.assertIn("status", str(context.exception))


def _build_manifest(
    *,
    source_dir: Path = Path("data/raw/papers"),
    chunk_size: int = 500,
    document_versions: dict[str, str] | None = None,
    embedding_batch_size: int = 8,
    embedding_timeout_seconds: float = 30.0,
    embedding_max_retries: int = 2,
    application_version: str = "test",
) -> IndexManifest:
    """创建测试使用的稳定 Manifest。"""

    return IndexManifest.build(
        source_dir=source_dir,
        chunker="CharacterChunker",
        chunk_size=chunk_size,
        chunk_overlap=80,
        embedding_provider="mock",
        embedding_model="mock-hash-embedding",
        embedding_dimension=16,
        embedding_batch_size=embedding_batch_size,
        embedding_timeout_seconds=embedding_timeout_seconds,
        embedding_max_retries=embedding_max_retries,
        vector_repository_type="local_json",
        vector_collection_name="papers",
        distance_metric="cosine",
        application_version=application_version,
        document_count=2,
        chunk_count=10,
        vector_count=10,
        document_versions=(
            document_versions if document_versions is not None else {"doc_1": "v_1"}
        ),
    )


if __name__ == "__main__":
    unittest.main()
