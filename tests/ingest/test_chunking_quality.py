"""chunking 质量检查测试。"""

from __future__ import annotations

import unittest

from app.core.models import DocumentChunk, ParsedDocument
from app.ingest.chunking.quality import ChunkingQualityChecker, ChunkingQualityConfig


def build_document(*, source_path: str = "paper.pdf") -> ParsedDocument:
    """构造测试用解析后文档。"""

    return ParsedDocument(
        doc_id="doc_test",
        content_hash="hash_test",
        version_id="v_test",
        title="测试论文",
        text="测试内容",
        source_path=source_path,
        metadata={"suffix": ".pdf"},
    )


def build_chunk(
    *,
    chunk_id: str = "chunk_test",
    doc_id: str = "doc_test",
    text: str = "测试 chunk 内容",
    source_path: str = "paper.pdf",
    token_count: int = 20,
    section: str | None = "Introduction",
    page_start: int | None = 1,
    metadata: dict | None = None,
) -> DocumentChunk:
    """构造测试用 chunk。"""

    return DocumentChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content_hash="hash_test",
        version_id="v_test",
        text=text,
        source_path=source_path,
        chunk_index=0,
        token_count=token_count,
        title="测试论文",
        section=section,
        page_start=page_start,
        page_end=page_start,
        metadata=metadata if metadata is not None else {"suffix": ".pdf"},
    )


class ChunkingQualityCheckerTest(unittest.TestCase):
    """验证 chunking 质量检查模块。"""

    def test_quality_checker_passes_for_healthy_chunks(self) -> None:
        result = ChunkingQualityChecker().check(
            documents=[build_document()],
            chunks=[build_chunk()],
            config=ChunkingQualityConfig(
                min_avg_token_count=1,
                max_avg_token_count=100,
                max_missing_pdf_page_ratio=0,
                max_missing_section_ratio=0,
            ),
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.issues, [])
        self.assertEqual(result.checked_document_count, 1)
        self.assertEqual(result.checked_chunk_count, 1)

    def test_quality_checker_reports_error_issues_for_required_quality_gates(self) -> None:
        bad_chunks = [
            build_chunk(
                chunk_id="chunk_empty",
                doc_id="",
                text=" ",
                source_path="",
                page_start=None,
                section=None,
                token_count=0,
                metadata={"suffix": ".pdf"},
            )
        ]

        result = ChunkingQualityChecker().check(
            documents=[build_document()],
            chunks=bad_chunks,
            config=ChunkingQualityConfig(max_missing_pdf_page_ratio=0, max_missing_section_ratio=0),
        )
        codes = {issue.code for issue in result.issues}

        self.assertFalse(result.passed)
        self.assertIn("missing_doc_id", codes)
        self.assertIn("missing_source_path", codes)
        self.assertIn("empty_chunk_found", codes)
        self.assertIn("missing_pdf_page_ratio_too_high", codes)
        self.assertGreaterEqual(result.error_count, 4)

    def test_missing_section_warning_does_not_fail_quality_check_by_default(self) -> None:
        result = ChunkingQualityChecker().check(
            documents=[build_document(source_path="note.md")],
            chunks=[
                build_chunk(
                    source_path="note.md",
                    section=None,
                    page_start=None,
                    metadata={"suffix": ".md"},
                )
            ],
            config=ChunkingQualityConfig(max_missing_section_ratio=0),
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.warning_count, 1)
        self.assertEqual(result.issues[0].code, "missing_section_ratio_too_high")

    def test_quality_checker_reports_no_chunks_created(self) -> None:
        result = ChunkingQualityChecker().check(
            documents=[build_document()],
            chunks=[],
            config=ChunkingQualityConfig(),
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.issues[0].code, "no_chunks_created")

    def test_quality_config_rejects_invalid_ratio(self) -> None:
        with self.assertRaises(ValueError) as context:
            ChunkingQualityConfig(max_missing_pdf_page_ratio=1.5)

        self.assertIn("max_missing_pdf_page_ratio", str(context.exception))


if __name__ == "__main__":
    unittest.main()
