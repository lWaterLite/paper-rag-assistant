"""配置管理测试。

运行方式：
python -m unittest discover -s tests
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from app.core.config import Settings
from app.core.errors import AppError, ErrorCode


class SettingsTest(unittest.TestCase):
    """验证 pydantic-settings 配置读取与校验。"""

    def test_settings_rejects_invalid_chunk_overlap(self) -> None:
        with self.assertRaises(ValidationError) as context:
            Settings(chunk_size=100, chunk_overlap=100)

        self.assertIn("chunk_overlap", str(context.exception))

    def test_settings_from_env_parses_bool_values(self) -> None:
        with patch.dict(os.environ, {"RAG_REQUIRE_CITATION": "off"}, clear=False):
            settings = Settings.from_env()

        self.assertFalse(settings.require_citation)

    def test_settings_from_env_rejects_invalid_int(self) -> None:
        with patch.dict(os.environ, {"RAG_TOP_K": "abc"}, clear=False):
            with self.assertRaises(AppError) as context:
                Settings.from_env()

        self.assertEqual(context.exception.code, ErrorCode.INVALID_CONFIG)
        self.assertIn("top_k", context.exception.message)

    def test_settings_accepts_supported_retrieval_strategy(self) -> None:
        settings = Settings(retrieval_strategy="bm25")

        self.assertEqual(settings.retrieval_strategy, "bm25")

    def test_settings_rejects_unsupported_retrieval_strategy(self) -> None:
        with self.assertRaises(ValidationError) as context:
            Settings(retrieval_strategy="random")

        self.assertIn("retrieval_strategy", str(context.exception))

    def test_settings_converts_index_storage_path_to_path(self) -> None:
        settings = Settings(index_storage_path="data/custom-index")

        self.assertEqual(settings.index_storage_path, Path("data/custom-index"))

    def test_settings_from_env_reads_added_fields(self) -> None:
        env = {
            "RAG_RETRIEVAL_STRATEGY": "hybrid",
            "RAG_INDEX_STORAGE_PATH": "data/env-index",
            "RAG_DEBUG_TRACE": "true",
        }
        with patch.dict(os.environ, env, clear=False):
            settings = Settings.from_env()

        self.assertEqual(settings.retrieval_strategy, "hybrid")
        self.assertEqual(settings.index_storage_path, Path("data/env-index"))
        self.assertTrue(settings.debug_trace)


if __name__ == "__main__":
    unittest.main()

