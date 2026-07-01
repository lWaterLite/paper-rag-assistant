"""结构化 metadata 测试。"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from app.core.metadata import BaseMetadata
from app.ingest.chunking.metadata import ChunkMetadata, ChunkMetadataBuilder


@dataclass(frozen=True)
class ExampleMetadata(BaseMetadata):
    """测试用 metadata 子类。"""

    name: str
    optional_value: int | None = None


class MetadataTest(unittest.TestCase):
    """验证通用 metadata 基类和 chunk metadata 构造器。"""

    def test_base_metadata_to_dict_excludes_none_by_default(self) -> None:
        metadata = ExampleMetadata(name="example")

        self.assertEqual(metadata.to_dict(), {"name": "example"})
        self.assertEqual(metadata.to_dict(exclude_none=False), {"name": "example", "optional_value": None})

    def test_chunk_metadata_builder_merges_document_standard_and_extra_metadata(self) -> None:
        metadata = ChunkMetadataBuilder().build(
            document_metadata={
                "filename": "paper.pdf",
                "chunker": "legacy",
                "token_start": 99,
            },
            chunk_metadata=ChunkMetadata(
                chunker="SectionAwareChunker",
                chunking_strategy="section_aware",
                chunk_size=600,
                chunk_overlap=100,
                tokenizer="char_approx",
                char_start=10,
                char_end=50,
                section_title="Introduction",
            ),
            extra_metadata={
                "token_start": 0,
                "token_end": 12,
                "ignored_none": None,
            },
        )

        self.assertEqual(metadata["filename"], "paper.pdf")
        self.assertEqual(metadata["chunker"], "SectionAwareChunker")
        self.assertEqual(metadata["chunking_strategy"], "section_aware")
        self.assertEqual(metadata["char_start"], 10)
        self.assertEqual(metadata["section_title"], "Introduction")
        self.assertEqual(metadata["token_start"], 0)
        self.assertEqual(metadata["token_end"], 12)
        self.assertNotIn("ignored_none", metadata)


if __name__ == "__main__":
    unittest.main()
