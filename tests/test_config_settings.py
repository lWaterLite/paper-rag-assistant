"""配置管理测试。

运行方式：
python -m unittest discover -s tests
"""

from __future__ import annotations

import os
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from app.core.settings import (
    ChunkingReportSettings,
    ChunkingSettings,
    EmbeddingSettings,
    EnvSettings,
    IndexBuilderSettings,
    IngestionReportSettings,
    PdfCleanerSettings,
    ProjectSettings,
    RetrievalSettings,
    VectorRepositorySettings,
)
from app.core.errors import AppError, ErrorCode


class SettingsTest(unittest.TestCase):
    """验证 EnvSettings 与 ProjectSettings 配置读取与校验。"""

    def test_settings_rejects_invalid_chunk_overlap(self) -> None:
        with self.assertRaises(ValidationError) as context:
            EnvSettings(chunk_size=100, chunk_overlap=100)

        self.assertIn("chunk_overlap", str(context.exception))

    def test_settings_rejects_chunk_overlap_greater_than_chunk_size(self) -> None:
        with self.assertRaises(ValidationError) as context:
            EnvSettings(chunk_size=100, chunk_overlap=120)

        self.assertIn("chunk_overlap", str(context.exception))

    def test_settings_rejects_non_positive_chunk_size(self) -> None:
        with self.assertRaises(ValidationError) as context:
            EnvSettings(chunk_size=0, chunk_overlap=0)

        self.assertIn("chunk_size", str(context.exception))

    def test_settings_from_env_wraps_invalid_chunk_window_as_app_error(self) -> None:
        env = {
            "RAG_CHUNK_SIZE": "100",
            "RAG_CHUNK_OVERLAP": "100",
        }
        with patch.dict(os.environ, env, clear=False):
            with self.assertRaises(AppError) as context:
                EnvSettings.from_env()

        self.assertEqual(context.exception.code, ErrorCode.INVALID_CONFIG)
        self.assertIn("chunk_overlap", context.exception.message)

    def test_settings_from_env_parses_bool_values(self) -> None:
        with patch.dict(os.environ, {"RAG_REQUIRE_CITATION": "off"}, clear=False):
            env_settings = EnvSettings.from_env()

        self.assertFalse(env_settings.require_citation)

    def test_settings_from_env_rejects_invalid_int(self) -> None:
        with patch.dict(os.environ, {"RAG_TOP_K": "abc"}, clear=False):
            with self.assertRaises(AppError) as context:
                EnvSettings.from_env()

        self.assertEqual(context.exception.code, ErrorCode.INVALID_CONFIG)
        self.assertIn("top_k", context.exception.message)

    def test_settings_accepts_supported_retrieval_strategy(self) -> None:
        env_settings = EnvSettings(retrieval_strategy="bm25")

        self.assertEqual(env_settings.retrieval_strategy, "bm25")

    def test_settings_rejects_unsupported_retrieval_strategy(self) -> None:
        with self.assertRaises(ValidationError) as context:
            EnvSettings(retrieval_strategy="random")

        self.assertIn("retrieval_strategy", str(context.exception))

    def test_settings_converts_index_storage_path_to_path(self) -> None:
        env_settings = EnvSettings(index_storage_path="data/custom-index")

        self.assertEqual(env_settings.index_storage_path, Path("data/custom-index"))

    def test_settings_from_env_reads_added_fields(self) -> None:
        env = {
            "RAG_RETRIEVAL_STRATEGY": "hybrid",
            "RAG_INDEX_STORAGE_PATH": "data/env-index",
            "RAG_DEBUG_TRACE": "true",
        }
        with patch.dict(os.environ, env, clear=False):
            env_settings = EnvSettings.from_env()

        self.assertEqual(env_settings.retrieval_strategy, "hybrid")
        self.assertEqual(env_settings.index_storage_path, Path("data/env-index"))
        self.assertTrue(env_settings.debug_trace)

    def test_settings_rejects_invalid_pdf_cleaner_ratio(self) -> None:
        with self.assertRaises(ValidationError) as context:
            PdfCleanerSettings(min_repeat_ratio=1.5)

        self.assertIn("min_repeat_ratio", str(context.exception))

    def test_settings_rejects_invalid_pdf_line_length_window(self) -> None:
        with self.assertRaises(ValidationError) as context:
            PdfCleanerSettings(min_line_length=20, max_line_length=10)

        self.assertIn("max_line_length", str(context.exception))

    def test_config_reads_structured_toml(self) -> None:
        config_path = Path(".tmp_config_tests") / f"config_{uuid.uuid4().hex}.toml"
        config_path.parent.mkdir(exist_ok=True)
        try:
            config_path.write_text(
                """
[loader]
recursive = false
ignored_dir_names = [".git", "__pycache__"]
temporary_file_suffixes = [".tmp"]

[pdf_cleaner]
edge_line_count = 3
min_repeat_ratio = 0.75
min_line_length = 4
max_line_length = 80

[ingestion_report]
output_dir = ".tmp_tests/ingestion-reports"

[chunking]
strategy = "fixed_token"
chunk_size = 256
chunk_overlap = 32
tokenizer = "simple_regex"

[chunking_report]
output_dir = ".tmp_tests/chunking-reports"

[embedding]
provider = "mock"
model = "mock-hash-embedding"
dimension = 24
batch_size = 16
timeout_seconds = 12.5
max_retries = 1
api_key_env_name = "TEST_OPENAI_API_KEY"

[vector_repository]
type = "local_json"
index_dir = ".tmp_tests/indexes"
collection_name = "test_collection"
distance_metric = "cosine"
persist = true

[index_builder]
manifest_filename = "test_manifest.json"
build_report_filename = "test_index_report.json"
skip_existing = false
fail_on_empty_chunk = false

[retrieval]
bm25_k1 = 1.8
bm25_b = 0.65
deduplicate_by_chunk_id = false
""".strip(),
                encoding="utf-8",
            )

            project_settings = ProjectSettings.from_toml(config_path)

            self.assertFalse(project_settings.loader.recursive)
            self.assertEqual(project_settings.loader.ignored_dir_names, frozenset({".git", "__pycache__"}))
            self.assertEqual(project_settings.loader.temporary_file_suffixes, (".tmp",))
            self.assertEqual(project_settings.pdf_cleaner.edge_line_count, 3)
            self.assertEqual(project_settings.pdf_cleaner.min_repeat_ratio, 0.75)
            self.assertEqual(project_settings.pdf_cleaner.max_line_length, 80)
            self.assertEqual(project_settings.ingestion_report.output_dir, Path(".tmp_tests/ingestion-reports"))
            self.assertEqual(project_settings.chunking.strategy, "fixed_token")
            self.assertEqual(project_settings.chunking.chunk_size, 256)
            self.assertEqual(project_settings.chunking.chunk_overlap, 32)
            self.assertEqual(project_settings.chunking.tokenizer, "simple_regex")
            self.assertEqual(project_settings.chunking_report.output_dir, Path(".tmp_tests/chunking-reports"))
            self.assertEqual(project_settings.embedding.dimension, 24)
            self.assertEqual(project_settings.embedding.batch_size, 16)
            self.assertEqual(project_settings.embedding.api_key_env_name, "TEST_OPENAI_API_KEY")
            self.assertEqual(project_settings.vector_repository.type, "local_json")
            self.assertEqual(project_settings.vector_repository.index_dir, Path(".tmp_tests/indexes"))
            self.assertEqual(project_settings.vector_repository.collection_name, "test_collection")
            self.assertTrue(project_settings.vector_repository.persist)
            self.assertEqual(project_settings.index_builder.manifest_filename, "test_manifest.json")
            self.assertEqual(project_settings.index_builder.build_report_filename, "test_index_report.json")
            self.assertFalse(project_settings.index_builder.skip_existing)
            self.assertFalse(project_settings.index_builder.fail_on_empty_chunk)
            self.assertEqual(project_settings.retrieval.bm25_k1, 1.8)
            self.assertEqual(project_settings.retrieval.bm25_b, 0.65)
            self.assertFalse(project_settings.retrieval.deduplicate_by_chunk_id)
        finally:
            if config_path.parent.exists():
                shutil.rmtree(config_path.parent, ignore_errors=True)

    def test_ingestion_report_settings_uses_logs_as_default_dir(self) -> None:
        settings = IngestionReportSettings()

        self.assertEqual(settings.output_dir, Path("logs"))

    def test_chunking_settings_rejects_invalid_overlap(self) -> None:
        with self.assertRaises(ValidationError) as context:
            ChunkingSettings(chunk_size=100, chunk_overlap=100)

        self.assertIn("chunk_overlap", str(context.exception))

    def test_chunking_settings_accepts_external_strategy_name(self) -> None:
        settings = ChunkingSettings(strategy=" semantic ")

        self.assertEqual(settings.strategy, "semantic")

    def test_chunking_settings_rejects_blank_strategy_name(self) -> None:
        with self.assertRaises(ValidationError) as context:
            ChunkingSettings(strategy=" ")

        self.assertIn("strategy", str(context.exception))

    def test_chunking_report_settings_uses_logs_as_default_dir(self) -> None:
        settings = ChunkingReportSettings()

        self.assertEqual(settings.output_dir, Path("logs"))

    def test_embedding_settings_rejects_invalid_batch_size(self) -> None:
        with self.assertRaises(ValidationError) as context:
            EmbeddingSettings(batch_size=0)

        self.assertIn("batch_size", str(context.exception))

    def test_vector_repository_settings_strips_collection_name(self) -> None:
        settings = VectorRepositorySettings(collection_name=" papers ")

        self.assertEqual(settings.collection_name, "papers")

    def test_vector_repository_settings_rejects_blank_collection_name(self) -> None:
        with self.assertRaises(ValidationError) as context:
            VectorRepositorySettings(collection_name=" ")

        self.assertIn("collection_name", str(context.exception))

    def test_indexing_settings_rejects_blank_manifest_filename(self) -> None:
        with self.assertRaises(ValidationError) as context:
            IndexBuilderSettings(manifest_filename=" ")

        self.assertIn("manifest_filename", str(context.exception))

    def test_retrieval_settings_rejects_invalid_bm25_parameters(self) -> None:
        with self.assertRaises(ValidationError) as context:
            RetrievalSettings(bm25_k1=0)

        self.assertIn("bm25_k1", str(context.exception))

        with self.assertRaises(ValidationError) as context:
            RetrievalSettings(bm25_b=1.5)

        self.assertIn("bm25_b", str(context.exception))


if __name__ == "__main__":
    unittest.main()
