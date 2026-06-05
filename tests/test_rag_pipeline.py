"""RAG 索引构建与在线问答流程测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

from app.core.config import Settings
from app.indexing.index_builder import IndexBuilder
from app.pipeline import RagPipeline


SAMPLE_SOURCE_DIR = Path("data/raw/papers")


class RagPipelineTest(unittest.TestCase):
    """验证离线索引流程与在线问答流程。"""

    def test_index_builder_builds_in_memory_index(self) -> None:
        settings = Settings(chunk_size=120, chunk_overlap=20, top_k=2)
        index, result = IndexBuilder(settings).build_from_directory(SAMPLE_SOURCE_DIR)

        self.assertGreater(result.document_count, 0)
        self.assertGreater(result.chunk_count, 0)
        self.assertEqual(result.chunk_count, result.vector_count)
        self.assertEqual(result.vector_count, index.vector_store.count())
        self.assertEqual(result.manifest.document_count, result.document_count)
        self.assertEqual(result.manifest.chunk_count, result.chunk_count)
        self.assertEqual(result.manifest.vector_count, result.vector_count)
        self.assertEqual(result.manifest.chunk_size, settings.chunk_size)
        self.assertEqual(result.manifest.chunk_overlap, settings.chunk_overlap)
        self.assertEqual(result.manifest.embedding_provider, "mock")
        self.assertGreater(len(result.manifest.document_versions), 0)

    def test_pipeline_returns_structured_answer(self) -> None:
        settings = Settings(chunk_size=120, chunk_overlap=20, top_k=2)
        index, _ = IndexBuilder(settings).build_from_directory(SAMPLE_SOURCE_DIR)
        answer = RagPipeline(settings=settings, index=index).ask("RAG 为什么需要引用？")

        self.assertTrue(answer.answer)
        self.assertTrue(answer.trace_id.startswith("trace_"))
        self.assertLessEqual(len(answer.retrieved_chunks), settings.top_k)
        self.assertGreater(len(answer.citations), 0)


if __name__ == "__main__":
    unittest.main()
