"""内存向量库测试。"""

from __future__ import annotations

import unittest
import shutil
import uuid
from pathlib import Path

from app.core.errors import AppError, ErrorCode
from app.core.models import DocumentChunk
from app.indexing.vector_store import InMemoryVectorStore, LocalJsonVectorStore


def build_chunk(chunk_id: str = "chunk_test", text: str = "测试 chunk") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_id="doc_test",
        content_hash="hash_test",
        version_id="v_test",
        text=text,
        source_path="test.md",
        chunk_index=0,
        token_count=len(text),
        title="测试文档",
    )


class InMemoryVectorStoreTest(unittest.TestCase):
    """验证向量写入、查询和维度校验。"""

    def test_add_sets_store_dimension_once(self) -> None:
        store = InMemoryVectorStore()

        store.add(build_chunk(), [1.0, 0.0, 0.0])

        self.assertEqual(store.dimension, 3)
        self.assertEqual(store.count(), 1)
        self.assertTrue(store.contains_chunk("chunk_test"))

    def test_add_ignores_duplicate_chunk_id(self) -> None:
        store = InMemoryVectorStore()

        store.add(build_chunk("chunk_same"), [1.0, 0.0])
        store.add(build_chunk("chunk_same"), [1.0, 0.0])

        self.assertEqual(store.count(), 1)

    def test_add_rejects_vector_with_different_dimension(self) -> None:
        store = InMemoryVectorStore()
        store.add(build_chunk(), [1.0, 0.0, 0.0])

        with self.assertRaises(AppError) as context:
            store.add(build_chunk("chunk_other"), [1.0, 0.0])

        self.assertEqual(context.exception.code, ErrorCode.INDEX_FAILED)
        self.assertIn("维度不一致", context.exception.message)

    def test_search_rejects_query_vector_with_different_dimension(self) -> None:
        store = InMemoryVectorStore()
        store.add(build_chunk(), [1.0, 0.0, 0.0])

        with self.assertRaises(AppError) as context:
            store.search([1.0, 0.0], top_k=1)

        self.assertEqual(context.exception.code, ErrorCode.RETRIEVAL_FAILED)
        self.assertIn("维度不一致", context.exception.message)

    def test_empty_store_search_returns_empty_results(self) -> None:
        store = InMemoryVectorStore()

        self.assertEqual(store.search([1.0, 0.0], top_k=3), [])

    def test_add_rejects_empty_vector(self) -> None:
        store = InMemoryVectorStore()

        with self.assertRaises(AppError) as context:
            store.add(build_chunk(), [])

        self.assertEqual(context.exception.code, ErrorCode.INDEX_FAILED)
        self.assertIn("不能为空", context.exception.message)

    def test_search_rejects_empty_query_vector_when_store_has_records(self) -> None:
        store = InMemoryVectorStore()
        store.add(build_chunk(), [1.0, 0.0])

        with self.assertRaises(AppError) as context:
            store.search([], top_k=1)

        self.assertEqual(context.exception.code, ErrorCode.RETRIEVAL_FAILED)
        self.assertIn("不能为空", context.exception.message)

    def test_store_dimension_is_not_shared_between_instances(self) -> None:
        first = InMemoryVectorStore()
        second = InMemoryVectorStore()

        first.add(build_chunk("chunk_first"), [1.0, 0.0, 0.0])
        second.add(build_chunk("chunk_second"), [1.0, 0.0])

        self.assertEqual(first.dimension, 3)
        self.assertEqual(second.dimension, 2)

    def test_local_json_vector_store_persists_and_loads_records(self) -> None:
        store_dir = Path(".tmp_tests") / f"vector_store_{uuid.uuid4().hex}"
        store_path = store_dir / "vector_store.json"
        store_dir.mkdir(parents=True, exist_ok=True)
        try:
            store = LocalJsonVectorStore(store_path)
            store.add(build_chunk("chunk_first", "RAG citation"), [1.0, 0.0])
            store.add(build_chunk("chunk_second", "BM25 retrieval"), [0.0, 1.0])
            store.persist()

            loaded_store = LocalJsonVectorStore(store_path)
            results = loaded_store.search([1.0, 0.0], top_k=1)

            self.assertEqual(loaded_store.count(), 2)
            self.assertEqual(loaded_store.dimension, 2)
            self.assertEqual(results[0].chunk_id, "chunk_first")
            self.assertEqual(results[0].metadata, {})
        finally:
            shutil.rmtree(store_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
