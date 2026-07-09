"""IndexBuilder 的 embedding cache 测试。"""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from app.core.errors import AppError, ErrorCode
from app.core.settings import (
    ChunkingReportSettings,
    ChunkingSettings,
    EnvSettings,
    IndexBuilderSettings,
    IngestionReportSettings,
    IngestionSettings,
    IndexingSettings,
    ProjectSettings,
    VectorRepositorySettings,
)
from app.factory import ApplicationFactory
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


class WrongDimensionEmbeddingClient:
    """故意返回错误维度，用来验证失败 manifest 写入。"""

    @property
    def provider(self) -> str:
        return "test"

    @property
    def model_name(self) -> str:
        return "wrong-dimension-embedding"

    @property
    def dimension(self) -> int:
        return 2

    def embed_text(self, text: str) -> list[float]:
        _ = text
        return [1.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


class IndexBuilderEmbeddingCacheTest(unittest.TestCase):
    """验证 IndexBuilder 会复用 embedding cache。"""

    def test_second_build_reuses_embedding_cache(self) -> None:
        env_settings = EnvSettings(chunk_size=120, chunk_overlap=20)
        cache = InMemoryEmbeddingCache()
        client = CountingEmbeddingClient()

        first_builder = create_index_builder(
            env_settings,
            ProjectSettings(),
            embedding_client=client,
            embedding_cache=cache,
        )
        _, first_result = first_builder.build_from_directory(Path("data/raw/papers"))

        second_builder = create_index_builder(
            env_settings,
            ProjectSettings(),
            embedding_client=client,
            embedding_cache=cache,
        )
        _, second_result = second_builder.build_from_directory(Path("data/raw/papers"))

        self.assertEqual(first_result.embedding_cache_hits, 0)
        self.assertEqual(first_result.embedding_cache_misses, first_result.chunk_count)
        self.assertEqual(second_result.embedding_cache_hits, second_result.chunk_count)
        self.assertEqual(second_result.embedding_cache_misses, 0)
        self.assertEqual(client.embedded_text_count, first_result.chunk_count)

    def test_same_builder_skips_chunks_already_in_vector_collection(self) -> None:
        env_settings = EnvSettings(chunk_size=120, chunk_overlap=20)
        client = CountingEmbeddingClient()
        builder = create_index_builder(env_settings, ProjectSettings(), embedding_client=client)

        _, first_result = builder.build_from_directory(Path("data/raw/papers"))
        index, second_result = builder.build_from_directory(Path("data/raw/papers"))

        self.assertEqual(second_result.skipped_existing_chunks, first_result.chunk_count)
        self.assertEqual(second_result.embedding_cache_hits, 0)
        self.assertEqual(second_result.embedding_cache_misses, 0)
        self.assertEqual(index.vector_collection.count(), first_result.chunk_count)
        self.assertEqual(client.embedded_text_count, first_result.chunk_count)

    def test_index_builder_writes_ingestion_report_from_project_settings(self) -> None:
        env_settings = EnvSettings(chunk_size=120, chunk_overlap=20)
        report_dir = Path(".tmp_tests") / f"ingestion_reports_{uuid.uuid4().hex}"
        chunking_report_dir = Path(".tmp_tests") / f"chunking_reports_{uuid.uuid4().hex}"
        project_settings = ProjectSettings(
            ingestion=IngestionSettings(
                report=IngestionReportSettings(output_dir=report_dir),
                chunking=ChunkingSettings(
                    report=ChunkingReportSettings(output_dir=chunking_report_dir)
                ),
            )
        )

        self.assertFalse(report_dir.exists())
        self.assertFalse(chunking_report_dir.exists())

        _, result = create_index_builder(env_settings, project_settings).build_from_directory(Path("data/raw/papers"))

        self.assertTrue(report_dir.exists())
        self.assertEqual(result.ingestion_report_path, report_dir / "ingestion_report.json")
        self.assertTrue(result.ingestion_report_path.exists())
        self.assertEqual(result.chunking_report_path, chunking_report_dir / "chunking_report.json")
        self.assertTrue(result.chunking_report_path.exists())

        report = json.loads(result.ingestion_report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["source_dir"], "data/raw/papers")
        self.assertEqual(report["succeeded"], result.document_count)
        self.assertEqual(report["failed"], len(result.ingestion_failures))
        self.assertTrue(report["trace_id"].startswith("trace_"))
        ingestion_stage = _find_trace_stage(result, "ingestion")
        self.assertEqual(ingestion_stage.detail["report_path"], result.ingestion_report_path.as_posix())
        self.assertEqual(report["trace"]["final_status"], "success")

        chunking_report = json.loads(result.chunking_report_path.read_text(encoding="utf-8"))
        self.assertEqual(chunking_report["chunk_count"], result.chunk_count)
        self.assertEqual(chunking_report["document_count"], result.document_count)
        chunking_stage = _find_trace_stage(result, "chunking")
        self.assertEqual(chunking_stage.detail["report_path"], result.chunking_report_path.as_posix())

    def test_index_builder_persists_local_json_index_artifacts(self) -> None:
        env_settings = EnvSettings(chunk_size=120, chunk_overlap=20)
        index_dir = Path(".tmp_tests") / f"indexes_{uuid.uuid4().hex}"
        report_dir = Path(".tmp_tests") / f"ingestion_reports_{uuid.uuid4().hex}"
        chunking_report_dir = Path(".tmp_tests") / f"chunking_reports_{uuid.uuid4().hex}"
        project_settings = ProjectSettings(
            indexing=IndexingSettings(
                vector_repository=VectorRepositorySettings(
                    type="local_json",
                    index_dir=index_dir,
                    collection_name="papers_test",
                    persist=True,
                ),
                builder=IndexBuilderSettings(
                    manifest_filename="manifest.json",
                    build_report_filename="index_build_report.json",
                ),
            ),
            ingestion=IngestionSettings(
                report=IngestionReportSettings(output_dir=report_dir),
                chunking=ChunkingSettings(
                    report=ChunkingReportSettings(output_dir=chunking_report_dir)
                ),
            ),
        )

        try:
            index, result = create_index_builder(env_settings, project_settings).build_from_directory(Path("data/raw/papers"))
            collection_dir = index_dir / "papers_test"

            self.assertEqual(index.vector_collection.count(), result.vector_count)
            self.assertEqual(result.manifest_path, collection_dir / "manifest.json")
            self.assertEqual(result.build_report_path, collection_dir / "index_build_report.json")
            self.assertTrue((collection_dir / "vector_collection.json").exists())
            self.assertTrue((collection_dir / "chunk_collection.json").exists())
            self.assertTrue((collection_dir / "document_collection.json").exists())
            self.assertTrue((collection_dir / "embedding_cache.json").exists())
            self.assertTrue(result.manifest_path.exists())
            self.assertTrue(result.build_report_path.exists())

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            build_report = json.loads(result.build_report_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "ready")
            self.assertEqual(manifest["vector_repository_type"], "local_json")
            self.assertEqual(manifest["vector_collection_name"], "papers_test")
            self.assertEqual(build_report["index_id"], result.manifest.index_id)
            self.assertEqual(build_report["vector_count"], result.vector_count)
            self.assertEqual(result.trace.stages[1].stage, "manifest_building")
            self.assertEqual(result.trace.stages[-1].stage, "manifest_ready")
        finally:
            shutil.rmtree(index_dir, ignore_errors=True)
            shutil.rmtree(report_dir, ignore_errors=True)
            shutil.rmtree(chunking_report_dir, ignore_errors=True)

    def test_index_builder_marks_manifest_failed_when_indexing_fails(self) -> None:
        index_dir = Path(".tmp_tests") / f"failed_index_{uuid.uuid4().hex}"
        report_dir = Path(".tmp_tests") / f"failed_ingestion_reports_{uuid.uuid4().hex}"
        chunking_report_dir = Path(".tmp_tests") / f"failed_chunking_reports_{uuid.uuid4().hex}"
        project_settings = ProjectSettings(
            indexing=IndexingSettings(
                vector_repository=VectorRepositorySettings(
                    type="local_json",
                    index_dir=index_dir,
                    collection_name="papers_test",
                    persist=True,
                ),
                builder=IndexBuilderSettings(
                    manifest_filename="manifest.json",
                    build_report_filename="index_build_report.json",
                ),
            ),
            ingestion=IngestionSettings(
                report=IngestionReportSettings(output_dir=report_dir),
                chunking=ChunkingSettings(
                    report=ChunkingReportSettings(output_dir=chunking_report_dir)
                ),
            ),
        )

        try:
            builder = create_index_builder(
                EnvSettings(chunk_size=120, chunk_overlap=20),
                project_settings,
                embedding_client=WrongDimensionEmbeddingClient(),
            )

            with self.assertRaises(AppError) as context:
                builder.build_from_directory(Path("data/raw/papers"))

            self.assertEqual(context.exception.code, ErrorCode.INDEX_FAILED)
            manifest_path = index_dir / "papers_test" / "manifest.json"
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertGreater(manifest["chunk_count"], 0)
            self.assertEqual(manifest["vector_count"], 0)
        finally:
            shutil.rmtree(index_dir, ignore_errors=True)
            shutil.rmtree(report_dir, ignore_errors=True)
            shutil.rmtree(chunking_report_dir, ignore_errors=True)

def _find_trace_stage(result, stage_name: str):
    """按阶段名查找 trace 记录，避免测试依赖固定下标。"""

    for stage in result.trace.stages:
        if stage.stage == stage_name:
            return stage
    raise AssertionError(f"没有找到 trace 阶段：{stage_name}")


def create_index_builder(env_settings: EnvSettings, project_settings: ProjectSettings, **overrides):
    """通过应用组合根创建测试用 IndexBuilder。"""

    return ApplicationFactory(
        env_settings=env_settings,
        project_settings=project_settings,
    ).build_index_builder(**overrides)


if __name__ == "__main__":
    unittest.main()
