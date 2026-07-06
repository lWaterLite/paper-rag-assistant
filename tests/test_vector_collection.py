"""向量集合与持久化 Repository 测试。"""

from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from app.core.errors import AppError, ErrorCode
from app.indexing.vector_collection import InMemoryVectorCollection, VectorRecord
from app.repositories.vector_repository import LocalJsonVectorRepository


def build_record(chunk_id: str = "chunk_test", vector: list[float] | None = None) -> VectorRecord:
    return VectorRecord(
        chunk_id=chunk_id,
        vector=[1.0, 0.0] if vector is None else vector,
        metadata={"doc_id": "doc_test"},
    )


class InMemoryVectorCollectionTest(unittest.TestCase):
    """验证向量集合写入、查询和维度校验。"""

    def test_add_sets_collection_dimension_once(self) -> None:
        collection = InMemoryVectorCollection()

        collection.add(build_record(vector=[1.0, 0.0, 0.0]))

        self.assertEqual(collection.dimension, 3)
        self.assertEqual(collection.count(), 1)
        self.assertTrue(collection.contains_chunk("chunk_test"))

    def test_add_ignores_duplicate_chunk_id(self) -> None:
        collection = InMemoryVectorCollection()

        collection.add(build_record("chunk_same"))
        collection.add(build_record("chunk_same"))

        self.assertEqual(collection.count(), 1)

    def test_add_rejects_vector_with_different_dimension(self) -> None:
        collection = InMemoryVectorCollection()
        collection.add(build_record(vector=[1.0, 0.0, 0.0]))

        with self.assertRaises(AppError) as context:
            collection.add(build_record("chunk_other", [1.0, 0.0]))

        self.assertEqual(context.exception.code, ErrorCode.INDEX_FAILED)
        self.assertIn("维度不一致", context.exception.message)

    def test_search_rejects_query_vector_with_different_dimension(self) -> None:
        collection = InMemoryVectorCollection()
        collection.add(build_record(vector=[1.0, 0.0, 0.0]))

        with self.assertRaises(AppError) as context:
            collection.search([1.0, 0.0], top_k=1)

        self.assertEqual(context.exception.code, ErrorCode.RETRIEVAL_FAILED)
        self.assertIn("维度不一致", context.exception.message)

    def test_empty_collection_search_returns_empty_results(self) -> None:
        collection = InMemoryVectorCollection()

        self.assertEqual(collection.search([1.0, 0.0], top_k=3), [])

    def test_add_rejects_empty_vector(self) -> None:
        collection = InMemoryVectorCollection()

        with self.assertRaises(AppError) as context:
            collection.add(build_record(vector=[]))

        self.assertEqual(context.exception.code, ErrorCode.INDEX_FAILED)
        self.assertIn("不能为空", context.exception.message)

    def test_search_rejects_empty_query_vector_when_collection_has_records(self) -> None:
        collection = InMemoryVectorCollection()
        collection.add(build_record())

        with self.assertRaises(AppError) as context:
            collection.search([], top_k=1)

        self.assertEqual(context.exception.code, ErrorCode.RETRIEVAL_FAILED)
        self.assertIn("不能为空", context.exception.message)

    def test_collection_dimension_is_not_shared_between_instances(self) -> None:
        first = InMemoryVectorCollection()
        second = InMemoryVectorCollection()

        first.add(build_record("chunk_first", [1.0, 0.0, 0.0]))
        second.add(build_record("chunk_second", [1.0, 0.0]))

        self.assertEqual(first.dimension, 3)
        self.assertEqual(second.dimension, 2)

    def test_local_json_vector_repository_persists_and_loads_collection(self) -> None:
        collection_dir = Path(".tmp_tests") / f"vector_collection_{uuid.uuid4().hex}"
        collection_path = collection_dir / "vector_collection.json"
        collection_dir.mkdir(parents=True, exist_ok=True)
        try:
            repository = LocalJsonVectorRepository(collection_path)
            collection = InMemoryVectorCollection()
            collection.add(build_record("chunk_first", [1.0, 0.0]))
            collection.add(build_record("chunk_second", [0.0, 1.0]))
            repository.save(collection)

            loaded_collection = repository.load()
            results = loaded_collection.search([1.0, 0.0], top_k=1)

            self.assertEqual(loaded_collection.count(), 2)
            self.assertEqual(loaded_collection.dimension, 2)
            self.assertEqual(results[0].chunk_id, "chunk_first")
            self.assertEqual(results[0].metadata, {"doc_id": "doc_test"})
        finally:
            shutil.rmtree(collection_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
