"""ContextPacker 测试。"""

from __future__ import annotations

import unittest

from app.core.models import RetrievedChunk
from app.retrieval.context_packer import SimpleContextPacker


def build_retrieved_chunk(
    chunk_id: str,
    text: str,
    *,
    doc_id: str = "doc_test",
    version_id: str = "v_test",
    chunk_index: int = 0,
    rank: int = 1,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content_hash=f"hash_{doc_id}",
        version_id=version_id,
        text=text,
        score=1.0,
        rank=rank,
        retriever="test",
        source_path=f"{doc_id}.md",
        chunk_index=chunk_index,
        title="测试文档",
        section="测试小节",
        metadata={"source": "test"},
    )


class SimpleContextPackerTest(unittest.TestCase):
    """验证上下文组织策略。"""

    def test_pack_deduplicates_same_text(self) -> None:
        chunks = [
            build_retrieved_chunk("chunk_1", "重复内容", chunk_index=0),
            build_retrieved_chunk("chunk_2", "重复内容", chunk_index=1),
        ]

        packed = SimpleContextPacker(max_context_chars=100).pack(chunks)

        self.assertEqual(len(packed.used_chunks), 1)
        self.assertEqual(len(packed.dropped_chunks), 1)
        self.assertEqual(packed.dropped_chunks[0].reason, "duplicate_content")

    def test_pack_merges_adjacent_chunks_from_same_document(self) -> None:
        chunks = [
            build_retrieved_chunk("chunk_1", "第一段", chunk_index=0),
            build_retrieved_chunk("chunk_2", "第二段", chunk_index=1),
        ]

        packed = SimpleContextPacker(max_context_chars=100).pack(chunks)

        self.assertEqual(len(packed.citations), 1)
        self.assertEqual(len(packed.used_chunks), 2)
        self.assertIn("第一段\n第二段", packed.context_text)

    def test_pack_drops_chunks_when_context_budget_is_full(self) -> None:
        chunks = [
            build_retrieved_chunk("chunk_1", "短内容", chunk_index=0),
            build_retrieved_chunk("chunk_2", "第二段内容很长", doc_id="doc_other", version_id="v_other", chunk_index=0),
            build_retrieved_chunk("chunk_3", "第三段内容", doc_id="doc_more", version_id="v_more", chunk_index=0),
        ]

        packed = SimpleContextPacker(max_context_chars=12).pack(chunks)

        self.assertEqual(len(packed.used_chunks), 1)
        self.assertEqual(len(packed.dropped_chunks), 2)
        self.assertEqual(packed.dropped_chunks[0].chunk_id, "chunk_2")
        self.assertEqual(packed.dropped_chunks[0].reason, "context_budget_exceeded")
        self.assertEqual(packed.dropped_chunks[1].chunk_id, "chunk_3")

    def test_pack_truncates_single_long_chunk_to_budget(self) -> None:
        chunks = [
            build_retrieved_chunk("chunk_1", "a" * 100, chunk_index=0),
        ]

        packed = SimpleContextPacker(max_context_chars=20).pack(chunks)

        self.assertEqual(len(packed.used_chunks), 1)
        self.assertLessEqual(len(packed.context_text), 20)
        self.assertTrue(packed.context_text.endswith("..."))

    def test_citation_ids_remain_sequential_after_deduplication(self) -> None:
        chunks = [
            build_retrieved_chunk("chunk_1", "重复内容", chunk_index=0),
            build_retrieved_chunk("chunk_2", "重复内容", chunk_index=1),
            build_retrieved_chunk("chunk_3", "新内容", doc_id="doc_other", version_id="v_other", chunk_index=0),
        ]

        packed = SimpleContextPacker(max_context_chars=100).pack(chunks)

        self.assertEqual([citation.citation_id for citation in packed.citations], ["C1", "C2"])
        self.assertEqual([chunk.chunk_id for chunk in packed.used_chunks], ["chunk_1", "chunk_3"])


if __name__ == "__main__":
    unittest.main()
