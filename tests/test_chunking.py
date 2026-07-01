"""chunking 子系统测试。"""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from app.core.models import ParsedDocument
from app.core.settings import ProjectSettings
from app.factory import build_configured_chunker
from app.ingest.chunkers import (
    CharacterChunker,
    ChunkerConfig,
    ChunkerRegistry,
    FixedTokenChunker,
    SectionAwareChunker,
    build_default_chunker_registry,
    estimate_token_count,
)
from app.ingest.chunking_report import ChunkingReportWriter


def build_document(text: str, *, source_path: str = "paper.md") -> ParsedDocument:
    """构造测试用解析后文档。"""

    return ParsedDocument(
        doc_id="doc_test",
        content_hash="hash_test",
        version_id="v_test",
        title="测试文档",
        text=text,
        source_path=source_path,
        metadata={"filename": Path(source_path).name, "suffix": Path(source_path).suffix},
    )


class ChunkingTest(unittest.TestCase):
    """验证 chunking 策略与报告输出。"""

    def test_default_registry_uses_configured_strategy(self) -> None:
        registry = build_default_chunker_registry()

        self.assertEqual(
            registry.list_strategies(),
            ("character", "fixed_token", "section_aware"),
        )
        self.assertIsInstance(registry.create(ChunkerConfig(strategy="character")), CharacterChunker)
        self.assertIsInstance(registry.create(ChunkerConfig(strategy="fixed_token")), FixedTokenChunker)
        self.assertIsInstance(registry.create(ChunkerConfig(strategy="section_aware")), SectionAwareChunker)

    def test_registry_rejects_duplicate_strategy(self) -> None:
        registry = ChunkerRegistry()
        registry.register("character", CharacterChunker)

        with self.assertRaises(ValueError) as context:
            registry.register("character", CharacterChunker)

        self.assertIn("已注册", str(context.exception))

    def test_factory_uses_injected_chunker_registry(self) -> None:
        registry = ChunkerRegistry()
        registry.register("section_aware", CustomSectionAwareChunker)

        chunker = build_configured_chunker(ProjectSettings(), chunker_registry=registry)

        self.assertIsInstance(chunker, CustomSectionAwareChunker)

    def test_fixed_token_chunker_splits_by_token_window(self) -> None:
        document = build_document("alpha beta gamma delta epsilon")
        chunker = FixedTokenChunker(
            ChunkerConfig(
                strategy="fixed_token",
                chunk_size=3,
                chunk_overlap=1,
                tokenizer="simple_regex",
            )
        )

        chunks = chunker.split(document)

        self.assertEqual([chunk.text for chunk in chunks], ["alpha beta gamma", "gamma delta epsilon"])
        self.assertEqual(chunks[0].metadata["token_start"], 0)
        self.assertEqual(chunks[0].metadata["token_end"], 3)
        self.assertEqual(chunks[1].metadata["token_start"], 2)
        self.assertEqual(chunks[1].metadata["token_end"], 5)

    def test_estimate_token_count_supports_simple_regex_tokenizer(self) -> None:
        self.assertEqual(estimate_token_count("RAG 系统 2024-06", "simple_regex"), 4)

    def test_simple_regex_tokenizer_keeps_common_acronym_decimal_and_date_tokens(self) -> None:
        text = "U.S.A. score is 3.14 on 2024-06."
        document = build_document(text)
        chunker = FixedTokenChunker(
            ChunkerConfig(
                strategy="fixed_token",
                chunk_size=1,
                chunk_overlap=0,
                tokenizer="simple_regex",
            )
        )
        tokens = [chunk.text for chunk in chunker.split(document)]

        self.assertEqual(tokens, ["U.S.A.", "score", "is", "3.14", "on", "2024-06", "."])

    def test_chunking_report_writer_outputs_quality_summary(self) -> None:
        document = build_document("# Intro\n内容", source_path="paper.pdf")
        config = ChunkerConfig(strategy="section_aware", chunk_size=100, chunk_overlap=10)
        chunks = SectionAwareChunker(config).split(document)
        tmp_dir = Path(".tmp_tests") / f"chunking_report_{uuid.uuid4().hex}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            output_path = Path(tmp_dir) / "chunking_report.json"
            result_path = ChunkingReportWriter().write(
                documents=[document],
                chunks=chunks,
                config=config,
                output_path=output_path,
            )

            report = json.loads(result_path.read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertEqual(report["strategy"], "section_aware")
        self.assertEqual(report["document_count"], 1)
        self.assertEqual(report["chunk_count"], len(chunks))
        self.assertEqual(report["documents"][0]["doc_id"], "doc_test")


class CustomSectionAwareChunker(SectionAwareChunker):
    """测试用外部 chunker，用于验证 registry 注入链路。"""


if __name__ == "__main__":
    unittest.main()
