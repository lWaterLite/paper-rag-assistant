"""纯文本解析与清洗测试。"""

from __future__ import annotations

import unittest

from app.core.models import RawDocument
from app.ingest.cleaners import BasicTextCleaner
from app.ingest.parsers import PlainTextParser


class PlainTextParserTest(unittest.TestCase):
    """验证 PlainTextParser 的文本清洗行为。"""

    def test_clean_text_normalizes_markdown_headings_and_blank_lines(self) -> None:
        raw_text = "#   RAG 入门   \n\n\n正文第一段。   \n\n\n##    Evaluation   \n正文第二段。"

        cleaned_text, metadata = PlainTextParser._clean_text(raw_text)

        self.assertEqual(
            cleaned_text,
            "# RAG 入门\n\n正文第一段。\n\n## Evaluation\n正文第二段。",
        )
        self.assertEqual(metadata["raw_text_length"], len(raw_text))
        self.assertEqual(metadata["cleaned_text_length"], len(cleaned_text))

    def test_clean_text_preserves_non_heading_indentation(self) -> None:
        raw_text = "普通段落\n    code line   "

        cleaned_text, _ = PlainTextParser._clean_text(raw_text)

        self.assertEqual(cleaned_text, "普通段落\n    code line")

    def test_parse_merges_cleaning_metadata(self) -> None:
        raw_text = "#   RAG 入门   \n\n\n正文。"
        document = RawDocument(
            doc_id="doc_test",
            source_path="test.md",
            file_type="md",
            content_hash="hash_test",
            version_id="v_test",
            raw_text=raw_text,
            metadata={"filename": "test.md"},
        )

        parsed = PlainTextParser(cleaner=BasicTextCleaner()).parse(document)

        self.assertEqual(parsed.text, "# RAG 入门\n\n正文。")
        self.assertEqual(parsed.metadata["filename"], "test.md")
        self.assertEqual(parsed.metadata["raw_text_length"], len(raw_text))
        self.assertEqual(parsed.metadata["cleaned_text_length"], len(parsed.text))


if __name__ == "__main__":
    unittest.main()
