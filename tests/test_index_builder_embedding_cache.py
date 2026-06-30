"""IndexBuilder 的 embedding cache 测试。"""

from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path

from app.core.settings import EnvSettings, IngestionReportSettings, ProjectSettings
from app.factory import build_index_builder
from app.indexing.embedding_cache import InMemoryEmbeddingCache


class CountingEmbeddingClient:
    """记录实际 embedding 调用文本数量的测试 client。"""

    def __init__(self, dimension: int = 2) -> None:
        self._dimension = dimension
        self.embedded_text_count = 0

    @property
    def provider(self) -> str:
        return "test"

    @property
    def model_name(self) -> str:
        return "counting-embedding"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        self.embedded_text_count += 1
        return [float(len(text)), 1.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.embedded_text_count += len(texts)
        return [[float(len(text)), 1.0] for text in texts]


class IndexBuilderEmbeddingCacheTest(unittest.TestCase):
    """验证 IndexBuilder 会复用 embedding cache。"""

    def test_second_build_reuses_embedding_cache(self) -> None:
        env_settings = EnvSettings(chunk_size=120, chunk_overlap=20)
        cache = InMemoryEmbeddingCache()
        client = CountingEmbeddingClient()

        first_builder = build_index_builder(env_settings, ProjectSettings(), embedding_client=client, embedding_cache=cache)
        _, first_result = first_builder.build_from_directory(Path("data/raw/papers"))

        second_builder = build_index_builder(env_settings, ProjectSettings(), embedding_client=client, embedding_cache=cache)
        _, second_result = second_builder.build_from_directory(Path("data/raw/papers"))

        self.assertEqual(first_result.embedding_cache_hits, 0)
        self.assertEqual(first_result.embedding_cache_misses, first_result.chunk_count)
        self.assertEqual(second_result.embedding_cache_hits, second_result.chunk_count)
        self.assertEqual(second_result.embedding_cache_misses, 0)
        self.assertEqual(client.embedded_text_count, first_result.chunk_count)

    def test_same_builder_skips_chunks_already_in_vector_store(self) -> None:
        env_settings = EnvSettings(chunk_size=120, chunk_overlap=20)
        client = CountingEmbeddingClient()
        builder = build_index_builder(env_settings, ProjectSettings(), embedding_client=client)

        _, first_result = builder.build_from_directory(Path("data/raw/papers"))
        index, second_result = builder.build_from_directory(Path("data/raw/papers"))

        self.assertEqual(second_result.skipped_existing_chunks, first_result.chunk_count)
        self.assertEqual(second_result.embedding_cache_hits, 0)
        self.assertEqual(second_result.embedding_cache_misses, 0)
        self.assertEqual(index.vector_store.count(), first_result.chunk_count)
        self.assertEqual(client.embedded_text_count, first_result.chunk_count)

    def test_index_builder_writes_ingestion_report_from_project_settings(self) -> None:
        env_settings = EnvSettings(chunk_size=120, chunk_overlap=20)
        report_dir = Path(".tmp_tests") / f"ingestion_reports_{uuid.uuid4().hex}"
        project_settings = ProjectSettings(
            ingestion_report=IngestionReportSettings(output_dir=report_dir)
        )

        self.assertFalse(report_dir.exists())

        _, result = build_index_builder(env_settings, project_settings).build_from_directory(Path("data/raw/papers"))

        self.assertTrue(report_dir.exists())
        self.assertEqual(result.ingestion_report_path, report_dir / "ingestion_report.json")
        self.assertTrue(result.ingestion_report_path.exists())

        report = json.loads(result.ingestion_report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["source_dir"], "data/raw/papers")
        self.assertEqual(report["succeeded"], result.document_count)
        self.assertEqual(report["failed"], len(result.ingestion_failures))
        self.assertTrue(report["trace_id"].startswith("trace_"))
        self.assertEqual(result.trace.stages[0].detail["report_path"], result.ingestion_report_path.as_posix())
        self.assertEqual(report["trace"]["final_status"], "success")


if __name__ == "__main__":
    unittest.main()
