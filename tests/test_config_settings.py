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

from app.core.settings import EnvSettings, PdfCleanerSettings, ProjectSettings
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
        finally:
            if config_path.parent.exists():
                shutil.rmtree(config_path.parent, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
