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
    BM25Settings,
    ChunkingReportSettings,
    ChunkingSettings,
    ContextPackingSettings,
    EmbeddingSettings,
    EnvSettings,
    HybridRetrievalSettings,
    IndexBuilderSettings,
    IngestionReportSettings,
    PdfCleanerSettings,
    ProjectSettings,
    RetrievalReportSettings,
    RerankingSettings,
    RetrievalSettings,
    TokenizerSettings,
    VectorRepositorySettings,
)


class SettingsTest(unittest.TestCase):
    """验证 EnvSettings 与 ProjectSettings 配置读取与校验。"""

    def test_env_settings_reads_openai_api_key_as_secret(self) -> None:
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-secret"},
            clear=False,
        ):
            env_settings = EnvSettings.from_env()

        self.assertEqual(
            env_settings.openai_api_key.get_secret_value(),
            "test-secret",
        )
        self.assertNotIn("test-secret", str(env_settings))

    def test_env_settings_treats_blank_secret_as_missing(self) -> None:
        settings = EnvSettings(OPENAI_API_KEY="")

        self.assertIsNone(settings.openai_api_key)

    def test_env_settings_rejects_legacy_non_secret_fields(self) -> None:
        with self.assertRaises(ValidationError) as context:
            EnvSettings(RAG_TOP_K="5")

        self.assertIn("RAG_TOP_K", str(context.exception))

    def test_retrieval_settings_accepts_external_strategy_name(self) -> None:
        settings = RetrievalSettings(strategy=" external ")

        self.assertEqual(settings.strategy, "external")

    def test_retrieval_settings_rejects_blank_strategy(self) -> None:
        with self.assertRaises(ValidationError) as context:
            RetrievalSettings(strategy=" ")

        self.assertIn("strategy", str(context.exception))

    def test_context_packing_settings_rejects_non_positive_budget(self) -> None:
        with self.assertRaises(ValidationError):
            ContextPackingSettings(max_context_tokens=0)

        with self.assertRaises(ValidationError):
            ContextPackingSettings(model_context_window=100, max_context_tokens=101)

    def test_reranking_settings_normalizes_strategy_and_validates_limits(self) -> None:
        settings = RerankingSettings(strategy=" lexical ", candidate_limit=10)

        self.assertEqual(settings.strategy, "lexical")
        with self.assertRaises(ValidationError):
            RerankingSettings(candidate_limit=0)

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
[ingestion.loader]
recursive = false
ignored_dir_names = [".git", "__pycache__"]
temporary_file_suffixes = [".tmp"]

[ingestion.cleaning.pdf]
edge_line_count = 3
min_repeat_ratio = 0.75
min_line_length = 4
max_line_length = 80

[ingestion.chunking]
strategy = "fixed_token"
chunk_size = 256
chunk_overlap = 32
tokenizer = "simple_regex"

[ingestion.chunking.report]
output_dir = ".tmp_tests/chunking-reports"

[ingestion.report]
output_dir = ".tmp_tests/ingestion-reports"

[indexing.embedding]
provider = "mock"
model = "mock-hash-embedding"
dimension = 24
batch_size = 16
timeout_seconds = 12.5
max_retries = 1

[indexing.vector_repository]
type = "local_json"
index_dir = ".tmp_tests/indexes"
collection_name = "test_collection"
distance_metric = "cosine"
persist = true

[indexing.builder]
manifest_filename = "test_manifest.json"
build_report_filename = "test_index_report.json"
skip_existing = false
fail_on_empty_chunk = false

[retrieval]
strategy = "hybrid"
top_k = 7
deduplicate_by_chunk_id = false

[retrieval.tokenizer]
strategy = "regex"

[retrieval.bm25]
k1 = 1.8
b = 0.65

[retrieval.hybrid]
candidate_multiplier = 4
rrf_rank_constant = 50
vector_weight = 1.2
bm25_weight = 0.8

[retrieval.reranking]
enabled = true
strategy = "lexical"
candidate_limit = 15
batch_size = 5
failure_mode = "fail_closed"

[retrieval.context_packing]
model_context_window = 8192
max_context_tokens = 2400
reserved_prompt_tokens = 300
reserved_output_tokens = 600
safety_margin_tokens = 80
max_chunks_per_document = 3

[retrieval.context_packing.token_estimator]
strategy = "regex"

[retrieval.report]
enabled = true
output_dir = ".tmp_tests/retrieval-reports"
include_result_text = true
result_preview_chars = 80
fail_on_write_error = true
""".strip(),
                encoding="utf-8",
            )

            project_settings = ProjectSettings.from_toml(config_path)

            self.assertFalse(project_settings.ingestion.loader.recursive)
            self.assertEqual(project_settings.ingestion.loader.ignored_dir_names, frozenset({".git", "__pycache__"}))
            self.assertEqual(project_settings.ingestion.loader.temporary_file_suffixes, (".tmp",))
            self.assertEqual(project_settings.ingestion.cleaning.pdf.edge_line_count, 3)
            self.assertEqual(project_settings.ingestion.cleaning.pdf.min_repeat_ratio, 0.75)
            self.assertEqual(project_settings.ingestion.cleaning.pdf.max_line_length, 80)
            self.assertEqual(project_settings.ingestion.report.output_dir, Path(".tmp_tests/ingestion-reports"))
            self.assertEqual(project_settings.ingestion.chunking.strategy, "fixed_token")
            self.assertEqual(project_settings.ingestion.chunking.chunk_size, 256)
            self.assertEqual(project_settings.ingestion.chunking.chunk_overlap, 32)
            self.assertEqual(project_settings.ingestion.chunking.tokenizer, "simple_regex")
            self.assertEqual(project_settings.ingestion.chunking.report.output_dir, Path(".tmp_tests/chunking-reports"))
            self.assertEqual(project_settings.indexing.embedding.dimension, 24)
            self.assertEqual(project_settings.indexing.embedding.batch_size, 16)
            self.assertEqual(project_settings.indexing.vector_repository.type, "local_json")
            self.assertEqual(project_settings.indexing.vector_repository.index_dir, Path(".tmp_tests/indexes"))
            self.assertEqual(project_settings.indexing.vector_repository.collection_name, "test_collection")
            self.assertTrue(project_settings.indexing.vector_repository.persist)
            self.assertEqual(project_settings.indexing.builder.manifest_filename, "test_manifest.json")
            self.assertEqual(project_settings.indexing.builder.build_report_filename, "test_index_report.json")
            self.assertFalse(project_settings.indexing.builder.skip_existing)
            self.assertFalse(project_settings.indexing.builder.fail_on_empty_chunk)
            self.assertEqual(project_settings.retrieval.bm25.k1, 1.8)
            self.assertEqual(project_settings.retrieval.bm25.b, 0.65)
            self.assertEqual(
                project_settings.retrieval.hybrid.candidate_multiplier, 4
            )
            self.assertEqual(
                project_settings.retrieval.hybrid.rrf_rank_constant, 50
            )
            self.assertEqual(project_settings.retrieval.hybrid.vector_weight, 1.2)
            self.assertEqual(project_settings.retrieval.hybrid.bm25_weight, 0.8)
            self.assertEqual(project_settings.retrieval.strategy, "hybrid")
            self.assertEqual(project_settings.retrieval.top_k, 7)
            self.assertEqual(
                project_settings.retrieval.context_packing.max_context_tokens,
                2400,
            )
            self.assertEqual(
                project_settings.retrieval.context_packing.model_context_window,
                8192,
            )
            self.assertEqual(
                project_settings.retrieval.context_packing.max_chunks_per_document,
                3,
            )
            self.assertEqual(
                project_settings.retrieval.context_packing.token_estimator.strategy,
                "regex",
            )
            self.assertTrue(project_settings.retrieval.reranking.enabled)
            self.assertEqual(project_settings.retrieval.reranking.candidate_limit, 15)
            self.assertEqual(project_settings.retrieval.reranking.failure_mode, "fail_closed")
            self.assertTrue(project_settings.retrieval.report.enabled)
            self.assertEqual(
                project_settings.retrieval.report.output_dir,
                Path(".tmp_tests/retrieval-reports"),
            )
            self.assertTrue(project_settings.retrieval.report.include_result_text)
            self.assertEqual(
                project_settings.retrieval.report.result_preview_chars,
                80,
            )
            self.assertTrue(project_settings.retrieval.report.fail_on_write_error)
            self.assertFalse(project_settings.retrieval.deduplicate_by_chunk_id)
            self.assertEqual(project_settings.retrieval.tokenizer.strategy, "regex")
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
            BM25Settings(k1=0)

        self.assertIn("k1", str(context.exception))

        with self.assertRaises(ValidationError) as context:
            BM25Settings(b=1.5)

        self.assertIn("b", str(context.exception))

    def test_tokenizer_settings_normalizes_strategy(self) -> None:
        settings = TokenizerSettings(strategy=" custom ")

        self.assertEqual(settings.strategy, "custom")

    def test_tokenizer_settings_rejects_blank_strategy(self) -> None:
        with self.assertRaises(ValidationError) as context:
            TokenizerSettings(strategy=" ")

        self.assertIn("strategy", str(context.exception))

    def test_hybrid_settings_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValidationError):
            HybridRetrievalSettings(candidate_multiplier=0)
        with self.assertRaises(ValidationError):
            HybridRetrievalSettings(rrf_rank_constant=0)
        with self.assertRaises(ValidationError):
            HybridRetrievalSettings(vector_weight=0)

    def test_retrieval_report_settings_rejects_invalid_preview_size(self) -> None:
        with self.assertRaises(ValidationError):
            RetrievalReportSettings(result_preview_chars=0)


if __name__ == "__main__":
    unittest.main()
