"""空文档处理测试。"""

from __future__ import annotations

import unittest

from app.core.config import Settings
from app.core.models import RawDocument
from app.ingest.chunkers import CharacterChunker, SectionAwareChunker
from app.ingest.parsers import PlainTextParser


def build_empty_raw_document() -> RawDocument:
    """构造一个空文档，模拟用户导入空文件的场景。"""

    return RawDocument(
        doc_id="doc_empty",
        source_path="empty.md",
        file_type="md",
        content_hash="hash_empty",
        version_id="v_empty",
        raw_text="",
        metadata={"filename": "empty.md"},
    )


class EmptyDocumentTest(unittest.TestCase):
    """验证空文档不会产生无意义 chunk。"""

    def test_parser_keeps_empty_text_and_uses_filename_as_title(self) -> None:
        parsed = PlainTextParser().parse(build_empty_raw_document())

        self.assertEqual(parsed.text, "")
        self.assertEqual(parsed.title, "empty.md")
        self.assertEqual(parsed.metadata["raw_text_length"], 0)
        self.assertEqual(parsed.metadata["cleaned_text_length"], 0)

    def test_character_chunker_returns_empty_list_for_empty_document(self) -> None:
        parsed = PlainTextParser().parse(build_empty_raw_document())
        chunks = CharacterChunker(Settings(chunk_size=100, chunk_overlap=10)).split(parsed)

        self.assertEqual(chunks, [])

    def test_section_aware_chunker_returns_empty_list_for_empty_document(self) -> None:
        parsed = PlainTextParser().parse(build_empty_raw_document())
        chunks = SectionAwareChunker(Settings(chunk_size=100, chunk_overlap=10)).split(parsed)

        self.assertEqual(chunks, [])


if __name__ == "__main__":
    unittest.main()
