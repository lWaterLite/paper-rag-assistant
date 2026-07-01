"""SectionAwareChunker 测试。"""

from __future__ import annotations

import unittest

from app.core.models import ParsedBlock, ParsedDocument
from app.ingest.chunking.strategies import ChunkerConfig, SectionAwareChunker


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
        chunks = SectionAwareChunker(_config(chunk_size=100, chunk_overlap=10)).split(build_document(text))

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].section, "Intro")
        self.assertEqual(chunks[1].section, "Eval")
        self.assertEqual(chunks[0].metadata["section_title"], "Intro")
        self.assertEqual(chunks[1].metadata["section_title"], "Eval")
        self.assertIn("# Intro", chunks[0].text)
        self.assertIn("## Eval", chunks[1].text)

    def test_long_section_is_split_again(self) -> None:
        text = "# Long\n" + "a" * 130
        chunks = SectionAwareChunker(_config(chunk_size=60, chunk_overlap=10)).split(build_document(text))

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.section == "Long" for chunk in chunks))
        self.assertTrue(all(len(chunk.text) <= 60 for chunk in chunks))

    def test_chunks_keep_document_title_and_version_fields(self) -> None:
        text = "# Intro\n第一节内容。"
        chunks = SectionAwareChunker(_config(chunk_size=100, chunk_overlap=10)).split(build_document(text))

        self.assertEqual(chunks[0].title, "测试文档")
        self.assertEqual(chunks[0].content_hash, "hash_test")
        self.assertEqual(chunks[0].version_id, "v_test")

    def test_char_offsets_point_to_original_text(self) -> None:
        text = "# Intro\n第一节内容。\n\n## Eval\n第二节内容。"
        chunks = SectionAwareChunker(_config(chunk_size=100, chunk_overlap=10)).split(build_document(text))

        for chunk in chunks:
            start = chunk.metadata["char_start"]
            end = chunk.metadata["char_end"]
            self.assertEqual(text[start:end], chunk.text)

    def test_split_keeps_page_metadata_from_parsed_blocks(self) -> None:
        document = ParsedDocument(
            doc_id="doc_pdf",
            content_hash="hash_pdf",
            version_id="v_pdf",
            title="论文 PDF",
            text="Introduction\n\nMethod",
            source_path="paper.pdf",
            metadata={"filename": "paper.pdf", "suffix": ".pdf"},
            blocks=[
                ParsedBlock(
                    block_id="block_intro",
                    doc_id="doc_pdf",
                    version_id="v_pdf",
                    text="Introduction",
                    block_type="heading",
                    source_path="paper.pdf",
                    page_start=1,
                    page_end=1,
                    section="Introduction",
                    char_start=0,
                    char_end=12,
                ),
                ParsedBlock(
                    block_id="block_method",
                    doc_id="doc_pdf",
                    version_id="v_pdf",
                    text="Method",
                    block_type="heading",
                    source_path="paper.pdf",
                    page_start=2,
                    page_end=2,
                    section="Method",
                    char_start=14,
                    char_end=20,
                ),
            ],
        )

        chunks = SectionAwareChunker(_config(chunk_size=100, chunk_overlap=10)).split(document)

        self.assertEqual([chunk.section for chunk in chunks], ["Introduction", "Method"])
        self.assertEqual([chunk.page_start for chunk in chunks], [1, 2])
        self.assertEqual([chunk.page_end for chunk in chunks], [1, 2])


def _config(*, chunk_size: int, chunk_overlap: int) -> ChunkerConfig:
    """构造测试用 section-aware chunker 配置。"""

    return ChunkerConfig(strategy="section_aware", chunk_size=chunk_size, chunk_overlap=chunk_overlap)


if __name__ == "__main__":
    unittest.main()
