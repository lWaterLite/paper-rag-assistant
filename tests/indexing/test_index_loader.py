"""已有索引加载测试。"""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from app.core.errors import AppError, ErrorCode
from app.core.settings import (
    EmbeddingSettings,
    EnvSettings,
    IndexingSettings,
    ProjectSettings,
    VectorRepositorySettings,
)
from app.factory import ApplicationFactory
from app.retrieval.retrievers.vector import VectorRetriever


class IndexLoaderTest(unittest.TestCase):
    """验证已有持久化索引可以被安全加载。"""

    def test_build_rag_index_from_storage_loads_local_json_index(self) -> None:
        index_dir = Path(".tmp_tests") / f"load_index_{uuid.uuid4().hex}"
        project_settings = ProjectSettings(
            indexing=IndexingSettings(
                vector_repository=VectorRepositorySettings(
                    type="local_json",
                    index_dir=index_dir,
                    collection_name="papers_test",
                )
            )
        )
        try:
            factory = ApplicationFactory(
                env_settings=EnvSettings(),
                project_settings=project_settings,
            )
            _, build_result = factory.build_index_builder().build_from_directory(Path("data/raw/papers"))

            loaded_index = factory.build_rag_index_from_storage()
            retriever = VectorRetriever(
                loaded_index.embedding_client,
                loaded_index.vector_collection,
                loaded_index.chunk_collection,
            )
            results = retriever.retrieve("RAG citation", top_k=2)

            self.assertEqual(loaded_index.manifest.index_id, build_result.manifest.index_id)
            self.assertEqual(loaded_index.vector_collection.count(), build_result.vector_count)
            self.assertLessEqual(len(results), 2)
            self.assertGreater(len(results), 0)
        finally:
            shutil.rmtree(index_dir, ignore_errors=True)

    def test_load_allows_embedding_execution_config_change(self) -> None:
        index_dir = Path(".tmp_tests") / f"load_execution_change_{uuid.uuid4().hex}"
        try:
            build_factory = _create_persistent_factory(
                index_dir,
                embedding_batch_size=4,
            )
            _, build_result = build_factory.build_index_builder().build_from_directory(
                Path("data/raw/papers")
            )
            load_factory = _create_persistent_factory(
                index_dir,
                embedding_batch_size=64,
            )

            loaded_index = load_factory.build_rag_index_from_storage()

            self.assertEqual(
                loaded_index.manifest.index_id,
                build_result.manifest.index_id,
            )
            self.assertEqual(
                loaded_index.manifest.build_provenance.embedding_batch_size,
                4,
            )
        finally:
            shutil.rmtree(index_dir, ignore_errors=True)

    def test_build_rag_index_from_storage_rejects_manifest_vector_count_mismatch(self) -> None:
        index_dir = Path(".tmp_tests") / f"broken_index_{uuid.uuid4().hex}"
        project_settings = ProjectSettings(
            indexing=IndexingSettings(
                vector_repository=VectorRepositorySettings(
                    type="local_json",
                    index_dir=index_dir,
                    collection_name="papers_test",
                )
            )
        )
        try:
            factory = ApplicationFactory(
                env_settings=EnvSettings(),
                project_settings=project_settings,
            )
            _, build_result = factory.build_index_builder().build_from_directory(Path("data/raw/papers"))
            manifest_path = build_result.manifest_path
            self.assertIsNotNone(manifest_path)
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_data["vector_count"] = manifest_data["vector_count"] + 1
            manifest_path.write_text(json.dumps(manifest_data, ensure_ascii=False, indent=2), encoding="utf-8")

            with self.assertRaises(AppError) as context:
                factory.build_rag_index_from_storage()

            self.assertEqual(context.exception.code, ErrorCode.INDEX_FAILED)
            self.assertIn("向量数量", context.exception.message)
        finally:
            shutil.rmtree(index_dir, ignore_errors=True)

    def test_build_rag_index_from_storage_rejects_manifest_mapping_type_mismatch(
        self,
    ) -> None:
        index_dir = Path(".tmp_tests") / f"invalid_manifest_{uuid.uuid4().hex}"
        try:
            factory = _create_persistent_factory(index_dir)
            _, build_result = factory.build_index_builder().build_from_directory(
                Path("data/raw/papers")
            )
            manifest_path = build_result.manifest_path
            self.assertIsNotNone(manifest_path)
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_data["artifact_definition"] = []
            manifest_path.write_text(
                json.dumps(manifest_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            with self.assertRaises(AppError) as context:
                factory.build_rag_index_from_storage()

            self.assertEqual(context.exception.code, ErrorCode.INVALID_CONFIG)
            self.assertIn("artifact_definition", context.exception.message)
        finally:
            shutil.rmtree(index_dir, ignore_errors=True)

    def test_build_rag_index_from_storage_rejects_vector_with_missing_chunk(self) -> None:
        index_dir = Path(".tmp_tests") / f"orphan_vector_{uuid.uuid4().hex}"
        try:
            factory = _create_persistent_factory(index_dir)
            factory.build_index_builder().build_from_directory(Path("data/raw/papers"))
            vector_path = index_dir / "papers_test" / "vector_collection.json"
            vector_data = json.loads(vector_path.read_text(encoding="utf-8"))
            vector_data["records"][0]["chunk_id"] = "chunk_orphaned_by_test"
            vector_path.write_text(
                json.dumps(vector_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            with self.assertRaises(AppError) as context:
                factory.build_rag_index_from_storage()

            self.assertEqual(context.exception.code, ErrorCode.INDEX_FAILED)
            self.assertIn("不存在的 Chunk", context.exception.message)
        finally:
            shutil.rmtree(index_dir, ignore_errors=True)

    def test_build_rag_index_from_storage_rejects_chunk_with_missing_document(self) -> None:
        index_dir = Path(".tmp_tests") / f"orphan_chunk_{uuid.uuid4().hex}"
        try:
            factory = _create_persistent_factory(index_dir)
            factory.build_index_builder().build_from_directory(Path("data/raw/papers"))
            chunk_path = index_dir / "papers_test" / "chunk_collection.json"
            chunk_data = json.loads(chunk_path.read_text(encoding="utf-8"))
            chunk_data["chunks"][0]["doc_id"] = "document_orphaned_by_test"
            chunk_path.write_text(
                json.dumps(chunk_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            with self.assertRaises(AppError) as context:
                factory.build_rag_index_from_storage()

            self.assertEqual(context.exception.code, ErrorCode.INDEX_FAILED)
            self.assertIn("不存在的解析文档", context.exception.message)
        finally:
            shutil.rmtree(index_dir, ignore_errors=True)

    def test_build_rag_index_from_storage_rejects_document_version_drift(self) -> None:
        index_dir = Path(".tmp_tests") / f"document_version_drift_{uuid.uuid4().hex}"
        try:
            factory = _create_persistent_factory(index_dir)
            factory.build_index_builder().build_from_directory(Path("data/raw/papers"))
            document_path = index_dir / "papers_test" / "document_collection.json"
            document_data = json.loads(document_path.read_text(encoding="utf-8"))
            document_data["raw_documents"][0]["version_id"] = "version_drifted_by_test"
            document_path.write_text(
                json.dumps(document_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            with self.assertRaises(AppError) as context:
                factory.build_rag_index_from_storage()

            self.assertEqual(context.exception.code, ErrorCode.INDEX_FAILED)
            self.assertIn("原始文档版本映射", context.exception.message)
        finally:
            shutil.rmtree(index_dir, ignore_errors=True)


def _create_persistent_factory(
    index_dir: Path,
    *,
    embedding_batch_size: int = 32,
) -> ApplicationFactory:
    """为单个测试创建隔离的本地 JSON 索引工厂。"""

    project_settings = ProjectSettings(
        indexing=IndexingSettings(
            embedding=EmbeddingSettings(batch_size=embedding_batch_size),
            vector_repository=VectorRepositorySettings(
                type="local_json",
                index_dir=index_dir,
                collection_name="papers_test",
            )
        )
    )
    return ApplicationFactory(
        env_settings=EnvSettings(),
        project_settings=project_settings,
    )

if __name__ == "__main__":
    unittest.main()
