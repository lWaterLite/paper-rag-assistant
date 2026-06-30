"""SectionAwareChunker 测试。"""

from __future__ import annotations

import unittest

from app.core.settings import EnvSettings
from app.core.models import ParsedDocument
from app.ingest.chunkers import SectionAwareChunker


def build_document(text: str) -> ParsedDocument:
    return ParsedDocument(
        doc_id="doc_test",
        content_hash="hash_test",
        version_id="v_test",
        title="测试文档",
        text=text,
        source_path="test.md",
        metadata={"filename": "test.md"},
    )


class SectionAwareChunkerTest(unittest.TestCase):
    """验证按小节切分和过长小节二次切分。"""

    def test_split_prefers_markdown_sections(self) -> None:
        text = "# Intro\n第一节内容。\n\n## Eval\n第二节内容。"
        chunks = SectionAwareChunker(EnvSettings(chunk_size=100, chunk_overlap=10)).split(build_document(text))

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].section, "Intro")
        self.assertEqual(chunks[1].section, "Eval")
        self.assertEqual(chunks[0].metadata["section_title"], "Intro")
        self.assertEqual(chunks[1].metadata["section_title"], "Eval")
        self.assertIn("# Intro", chunks[0].text)
        self.assertIn("## Eval", chunks[1].text)

    def test_long_section_is_split_again(self) -> None:
        text = "# Long\n" + "a" * 130
        chunks = SectionAwareChunker(EnvSettings(chunk_size=60, chunk_overlap=10)).split(build_document(text))

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.section == "Long" for chunk in chunks))
        self.assertTrue(all(len(chunk.text) <= 60 for chunk in chunks))

    def test_chunks_keep_document_title_and_version_fields(self) -> None:
        text = "# Intro\n第一节内容。"
        chunks = SectionAwareChunker(EnvSettings(chunk_size=100, chunk_overlap=10)).split(build_document(text))

        self.assertEqual(chunks[0].title, "测试文档")
        self.assertEqual(chunks[0].content_hash, "hash_test")
        self.assertEqual(chunks[0].version_id, "v_test")

    def test_char_offsets_point_to_original_text(self) -> None:
        text = "# Intro\n第一节内容。\n\n## Eval\n第二节内容。"
        chunks = SectionAwareChunker(EnvSettings(chunk_size=100, chunk_overlap=10)).split(build_document(text))

        for chunk in chunks:
            start = chunk.metadata["char_start"]
            end = chunk.metadata["char_end"]
            self.assertEqual(text[start:end], chunk.text)


if __name__ == "__main__":
    unittest.main()

