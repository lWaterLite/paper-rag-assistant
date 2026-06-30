"""文档身份与版本信息测试。"""

from __future__ import annotations

import unittest
import uuid
from pathlib import Path

from app.core.settings import EnvSettings, ProjectSettings
from app.core.errors import AppError, ErrorCode
from app.factory import build_local_text_loader
from app.ingest.chunkers import CharacterChunker
from app.ingest.cleaners import BasicTextCleaner
from app.ingest.parsers import PlainTextParser


SAMPLE_DOCUMENT = Path("data/raw/papers/rag_intro_note.md")


class DocumentIdentityTest(unittest.TestCase):
    """验证 doc_id、content_hash、version_id 和 chunk_id 的行为。"""

    def test_loader_rejects_missing_directory(self) -> None:
        missing_dir = Path("tests") / f"missing_{uuid.uuid4().hex}"

        with self.assertRaises(AppError) as context:
            build_local_text_loader(ProjectSettings()).load_directory(missing_dir)

        self.assertEqual(context.exception.code, ErrorCode.DOCUMENT_LOAD_FAILED)
        self.assertIn("文档目录不存在", context.exception.message)

    def test_loader_keeps_stable_doc_id_and_changes_version_when_content_changes(self) -> None:
        loader = build_local_text_loader(ProjectSettings())
        first = loader.load_file(SAMPLE_DOCUMENT)
        second = loader.load_file(SAMPLE_DOCUMENT)

        self.assertEqual(first.doc_id, second.doc_id)
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertEqual(first.version_id, second.version_id)

        changed_content_hash = loader._build_content_hash("changed content")
        changed_version_id = loader._build_version_id(first.doc_id, changed_content_hash)

        self.assertNotEqual(first.content_hash, changed_content_hash)
        self.assertNotEqual(first.version_id, changed_version_id)

    def test_parser_and_chunker_preserve_document_version_fields(self) -> None:
        raw_document = build_local_text_loader(ProjectSettings()).load_file(SAMPLE_DOCUMENT)
        parsed_document = PlainTextParser(cleaner=BasicTextCleaner()).parse(raw_document)
        chunks = CharacterChunker(EnvSettings(chunk_size=120, chunk_overlap=20)).split(parsed_document)

        self.assertEqual(parsed_document.content_hash, raw_document.content_hash)
        self.assertEqual(parsed_document.version_id, raw_document.version_id)
        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0].content_hash, raw_document.content_hash)
        self.assertEqual(chunks[0].version_id, raw_document.version_id)

    def test_chunk_id_is_stable_for_same_document_version(self) -> None:
        env_settings = EnvSettings(chunk_size=80, chunk_overlap=10)
        loader = build_local_text_loader(ProjectSettings())
        parser = PlainTextParser(cleaner=BasicTextCleaner())
        chunker = CharacterChunker(env_settings)

        first_chunks = chunker.split(parser.parse(loader.load_file(SAMPLE_DOCUMENT)))
        second_chunks = chunker.split(parser.parse(loader.load_file(SAMPLE_DOCUMENT)))

        self.assertEqual([chunk.chunk_id for chunk in first_chunks], [chunk.chunk_id for chunk in second_chunks])


if __name__ == "__main__":
    unittest.main()
