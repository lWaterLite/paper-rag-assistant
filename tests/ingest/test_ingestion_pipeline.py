"""子模块 2 ingestion pipeline 测试。"""

from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path

from app.core.settings import IngestionSettings, LoaderSettings, ProjectSettings
from app.core.errors import AppError, ErrorCode
from app.ingest.models import RawDocument
from app.factory import ApplicationFactory
from app.ingest.loading import (
    DocumentIdentityBuilder,
    LocalDocumentLoader,
    LocalDocumentLoaderConfig,
)
from app.ingest.parsing import (
    BasicTextCleaner,
    HtmlDocumentParser,
    HtmlTextCleaner,
    MarkdownParser,
    ParserRegistry,
    PdfDocumentParser,
    PdfTextCleaner,
    PdfTextCleanerConfig,
)
from app.ingest.pipeline import IngestionPipeline
from app.ingest.reporting import IngestionReportWriter


class IngestionPipelineTest(unittest.TestCase):
    """验证真实文档加载、解析和失败隔离。"""

    def test_loader_reads_markdown_html_and_keeps_raw_bytes(self) -> None:
        documents = create_local_document_loader(ProjectSettings()).load_directory(Path("data/raw/papers"))

        self.assertIn("markdown", {document.file_type for document in documents})
        self.assertIn("html", {document.file_type for document in documents})
        self.assertTrue(all(document.raw_bytes for document in documents))
        self.assertTrue(all(document.content_hash for document in documents))
        self.assertTrue(all(document.version_id.startswith("v_") for document in documents))

    def test_markdown_parser_extracts_frontmatter_and_section_blocks(self) -> None:
        raw = RawDocument(
            doc_id="doc_md",
            source_path="note.md",
            file_type="markdown",
            content_hash="hash_md",
            version_id="v_md",
            raw_text='---\ntitle: "RAG Evaluation"\nauthors: Alice\n---\n# Intro\n\nRAG needs citations.\n\n## Metrics\n\nMRR matters.',
            metadata={"filename": "note.md"},
        )

        parsed = MarkdownParser(cleaner=BasicTextCleaner()).parse(raw)

        self.assertEqual(parsed.title, "RAG Evaluation")
        self.assertEqual(parsed.metadata["frontmatter_authors"], "Alice")
        self.assertIn("RAG needs citations.", parsed.text)
        self.assertTrue(any(block.block_type == "heading" for block in parsed.blocks))
        self.assertTrue(any(block.section == "Metrics" for block in parsed.blocks))

    def test_html_parser_removes_navigation_and_keeps_metadata(self) -> None:
        raw = RawDocument(
            doc_id="doc_html",
            source_path="page.html",
            file_type="html",
            content_hash="hash_html",
            version_id="v_html",
            raw_text="""
            <html>
              <head>
                <title>RAG Project Page</title>
                <meta name="description" content="Project description">
                <link rel="canonical" href="https://example.org/rag">
              </head>
              <body>
                <nav>Home About</nav>
                <main><article><h1>RAG Project Page</h1><p>Faithfulness requires citations.</p></article></main>
                <footer>Footer text</footer>
              </body>
            </html>
            """,
            metadata={"filename": "page.html"},
        )

        parsed = HtmlDocumentParser(cleaner=HtmlTextCleaner()).parse(raw)

        self.assertEqual(parsed.title, "RAG Project Page")
        self.assertEqual(parsed.metadata["description"], "Project description")
        self.assertEqual(parsed.metadata["canonical_url"], "https://example.org/rag")
        self.assertIn("Faithfulness requires citations.", parsed.text)
        self.assertNotIn("Home About", parsed.text)
        self.assertNotIn("Footer text", parsed.text)

    def test_ingestion_pipeline_records_bad_file_without_dropping_good_files(self) -> None:
        result = IngestionPipeline(
            loader=FakeMixedLoader(),
            parser_registry=ParserRegistry(parsers=[MarkdownParser(cleaner=BasicTextCleaner())]),
        ).ingest_directory(Path("data/raw/papers"))

        self.assertEqual(len(result.documents), 1)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.documents[0].parsed_document.title, "Good Paper")
        self.assertEqual(result.failures[0].stage, "loading")
        self.assertEqual(result.failures[0].error_code, ErrorCode.DOCUMENT_LOAD_FAILED.value)

    def test_pdf_parser_reports_clear_error_for_invalid_pdf(self) -> None:
        raw = RawDocument(
            doc_id="doc_pdf",
            source_path="broken.pdf",
            file_type="pdf",
            content_hash="hash_pdf",
            version_id="v_pdf",
            raw_text="",
            raw_bytes=b"this is not a valid pdf",
            metadata={"filename": "broken.pdf"},
        )

        with self.assertRaises(AppError) as context:
            PdfDocumentParser(cleaner=PdfTextCleaner(config=PdfTextCleanerConfig())).parse(raw)

        self.assertEqual(context.exception.code, ErrorCode.DOCUMENT_PARSE_FAILED)

    def test_factory_applies_non_recursive_loader_config(self) -> None:
        project_settings = ProjectSettings(
            ingestion=IngestionSettings(loader=LoaderSettings(recursive=False))
        )
        loader = create_local_document_loader(project_settings)

        paths = list(loader.iter_supported_files(Path("data/raw/papers")))

        self.assertTrue(paths)
        self.assertTrue(all(path.parent == Path("data/raw/papers") for path in paths))

    def test_loader_skips_hidden_paths_ignored_dirs_relative_paths_and_temp_files(self) -> None:
        loader = LocalDocumentLoader(
            config=LocalDocumentLoaderConfig(
                recursive=True,
                ignored_dir_names=frozenset({"__pycache__"}),
                ignored_relative_paths=("ignored/indexes",),
                skip_hidden_paths=True,
                temporary_file_prefixes=("~$",),
                temporary_file_suffixes=(".tmp",),
            ),
            identity_builder=DocumentIdentityBuilder(),
        )
        source_dir = Path("data/raw/papers")
        paths = [
            source_dir / "visible.md",
            source_dir / "__pycache__" / "cached.md",
            source_dir / ".hidden" / "draft.md",
            source_dir / "ignored" / "indexes" / "manifest.md",
            source_dir / "~$office.md",
            source_dir / "download.tmp",
        ]

        kept_paths = [path for path in paths if not loader._should_skip_path(path, source_dir)]

        self.assertEqual(kept_paths, [source_dir / "visible.md"])

    def test_pdf_cleaner_config_controls_repeated_edge_detection(self) -> None:
        pages = [
            (1, "Conference Header\nBody A\nConference Header"),
            (2, "Conference Header\nBody B\nConference Header"),
            (3, "Different Header\nBody C\nDifferent Header"),
        ]
        strict_cleaner = PdfTextCleaner(
            config=PdfTextCleanerConfig(edge_line_count=1, min_repeat_ratio=1.0)
        )
        lenient_cleaner = PdfTextCleaner(
            config=PdfTextCleanerConfig(edge_line_count=1, min_repeat_ratio=0.6)
        )

        self.assertEqual(strict_cleaner._detect_repeated_edge_lines(pages), set())
        self.assertEqual(
            lenient_cleaner._detect_repeated_edge_lines(pages),
            {"Conference Header"},
        )

    def test_ingestion_report_writer_writes_json_report(self) -> None:
        result = IngestionPipeline(
            loader=FakeMixedLoader(),
            parser_registry=ParserRegistry(parsers=[MarkdownParser(cleaner=BasicTextCleaner())]),
        ).ingest_directory(Path("data/raw/papers"))

        output_path = Path(".tmp_tests") / f"ingestion_report_{uuid.uuid4().hex}.json"
        output_path.parent.mkdir(exist_ok=True)
        written_path = IngestionReportWriter().write(result, output_path)

        report = json.loads(written_path.read_text(encoding="utf-8"))

        self.assertEqual(written_path, output_path)
        self.assertEqual(report["trace_id"], result.trace.trace_id)
        self.assertEqual(report["source_dir"], "data/raw/papers")
        self.assertEqual(report["succeeded"], 1)
        self.assertEqual(report["failed"], 1)
        self.assertFalse(report["success"])
        self.assertEqual(report["documents"][0]["doc_id"], "doc_good")
        self.assertEqual(report["documents"][0]["title"], "Good Paper")
        self.assertEqual(report["documents"][0]["block_count"], 2)
        self.assertEqual(report["failures"][0]["stage"], "loading")
        self.assertEqual(report["trace"]["final_status"], "success")


class FakeMixedLoader:
    """用于模拟一个好文件和一个坏文件。"""

    supported_suffixes = {".md", ".txt"}

    def load_directory(self, source_dir: Path) -> list[RawDocument]:
        return [self.load_file(path) for path in self.iter_supported_files(source_dir)]

    def iter_supported_files(self, source_dir: Path):
        yield source_dir / "good.md"
        yield source_dir / "bad.txt"

    def load_file(self, path: Path) -> RawDocument:
        if path.name == "bad.txt":
            raise AppError(ErrorCode.DOCUMENT_LOAD_FAILED, f"文件读取失败：{path}")
        return RawDocument(
            doc_id="doc_good",
            source_path=str(path),
            file_type="markdown",
            content_hash="hash_good",
            version_id="v_good",
            raw_text="# Good Paper\n\nRAG uses retrieval.",
            raw_bytes=b"# Good Paper\n\nRAG uses retrieval.",
            metadata={"filename": "good.md"},
        )


def create_local_document_loader(project_settings: ProjectSettings):
    """通过应用组合根创建测试用 loader。"""

    return ApplicationFactory(project_settings=project_settings).ingestion.build_local_document_loader()


if __name__ == "__main__":
    unittest.main()
