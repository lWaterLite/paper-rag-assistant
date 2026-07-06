"""已有索引加载测试。"""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from app.core.errors import AppError, ErrorCode
from app.core.settings import EnvSettings, ProjectSettings, VectorRepositorySettings
from app.factory import build_index_builder, build_rag_index_from_storage
from app.retrieval.retrievers import VectorRetriever


class IndexLoaderTest(unittest.TestCase):
    """验证已有持久化索引可以被安全加载。"""

    def test_build_rag_index_from_storage_loads_local_json_index(self) -> None:
        index_dir = Path(".tmp_tests") / f"load_index_{uuid.uuid4().hex}"
        project_settings = ProjectSettings(
            vector_repository=VectorRepositorySettings(
                type="local_json",
                index_dir=index_dir,
                collection_name="papers_test",
                persist=True,
            )
        )
        try:
            _, build_result = build_index_builder(
                EnvSettings(chunk_size=120, chunk_overlap=20),
                project_settings,
            ).build_from_directory(Path("data/raw/papers"))

            loaded_index = build_rag_index_from_storage(project_settings)
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

    def test_build_rag_index_from_storage_rejects_manifest_vector_count_mismatch(self) -> None:
        index_dir = Path(".tmp_tests") / f"broken_index_{uuid.uuid4().hex}"
        project_settings = ProjectSettings(
            vector_repository=VectorRepositorySettings(
                type="local_json",
                index_dir=index_dir,
                collection_name="papers_test",
                persist=True,
            )
        )
        try:
            _, build_result = build_index_builder(
                EnvSettings(chunk_size=120, chunk_overlap=20),
                project_settings,
            ).build_from_directory(Path("data/raw/papers"))
            manifest_path = build_result.manifest_path
            self.assertIsNotNone(manifest_path)
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_data["vector_count"] = manifest_data["vector_count"] + 1
            manifest_path.write_text(json.dumps(manifest_data, ensure_ascii=False, indent=2), encoding="utf-8")

            with self.assertRaises(AppError) as context:
                build_rag_index_from_storage(project_settings)

            self.assertEqual(context.exception.code, ErrorCode.INDEX_FAILED)
            self.assertIn("向量数量", context.exception.message)
        finally:
            shutil.rmtree(index_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
