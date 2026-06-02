"""子模块 1 脚手架自检。

这些测试使用 Python 标准库 unittest，不要求你安装 pytest。
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
from app.indexing.index_builder import IndexBuilder
from app.pipeline import RagPipeline


class SubmoduleOneScaffoldTest(unittest.TestCase):
    """验证练习脚手架的关键数据流。"""

    def test_index_builder_builds_in_memory_index(self) -> None:
        settings = Settings(chunk_size=120, chunk_overlap=20, top_k=2)
        index, result = IndexBuilder(settings).build_from_directory(Path("data/raw/papers"))

        self.assertGreater(result.document_count, 0)
        self.assertGreater(result.chunk_count, 0)
        self.assertEqual(result.chunk_count, result.vector_count)
        self.assertEqual(result.vector_count, index.vector_store.count())

    def test_pipeline_returns_structured_answer(self) -> None:
        settings = Settings(chunk_size=120, chunk_overlap=20, top_k=2)
        index, _ = IndexBuilder(settings).build_from_directory(Path("data/raw/papers"))
        answer = RagPipeline(settings=settings, index=index).ask("RAG 为什么需要引用？")

        self.assertTrue(answer.answer)
        self.assertTrue(answer.trace_id.startswith("trace_"))
        self.assertLessEqual(len(answer.retrieved_chunks), settings.top_k)
        self.assertGreater(len(answer.citations), 0)

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

    # TODO 练习 14：
    # 请你继续补充测试：
    # 1. source 目录不存在时是否抛出清晰错误。
    # 2. chunk_overlap >= chunk_size 时是否抛出错误。
    # 3. max_context_chars 很小时，context packer 是否会丢弃后续 chunk。
    # 4. 空文档是否不会生成 chunk。


if __name__ == "__main__":
    unittest.main()
