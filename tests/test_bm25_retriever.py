"""BM25Retriever 测试。"""

from __future__ import annotations

import unittest

from app.core.models import DocumentChunk
from app.ingest.chunking.collection import InMemoryChunkCollection
from app.retrieval.retrievers import BM25Retriever


def build_chunk(chunk_id: str, text: str, section: str | None = None) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        content_hash=f"hash_{chunk_id}",
        version_id=f"v_{chunk_id}",
        text=text,
        source_path=f"{chunk_id}.md",
        chunk_index=0,
        token_count=len(text),
        title=f"文档 {chunk_id}",
        section=section,
        metadata={"source": "test"},
    )


class BM25RetrieverTest(unittest.TestCase):
    """验证 BM25 关键词检索。"""

    def test_retrieve_returns_ranked_retrieved_chunks(self) -> None:
        chunks = [
            build_chunk("a", "RAG evaluation includes faithfulness.", "Evaluation"),
            build_chunk("b", "Vector databases store embedding vectors.", "Indexing"),
        ]
        results = BM25Retriever(chunks).retrieve("faithfulness evaluation", top_k=2)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "a")
        self.assertEqual(results[0].retriever, "bm25")
        self.assertEqual(results[0].rank, 1)
        self.assertGreater(results[0].score, 0)

    def test_retrieve_preserves_chunk_metadata(self) -> None:
        chunk = build_chunk("a", "RAG evaluation includes faithfulness.", "Evaluation")
        result = BM25Retriever([chunk]).retrieve("faithfulness", top_k=1)[0]

        self.assertEqual(result.doc_id, chunk.doc_id)
        self.assertEqual(result.content_hash, chunk.content_hash)
        self.assertEqual(result.version_id, chunk.version_id)
        self.assertEqual(result.source_path, chunk.source_path)
        self.assertEqual(result.title, chunk.title)
        self.assertEqual(result.section, chunk.section)
        self.assertEqual(result.metadata, chunk.metadata)

    def test_retrieve_respects_top_k(self) -> None:
        chunks = [
            build_chunk("a", "RAG evaluation faithfulness."),
            build_chunk("b", "RAG evaluation relevance."),
            build_chunk("c", "RAG evaluation precision."),
        ]
        results = BM25Retriever(chunks).retrieve("RAG evaluation", top_k=2)

        self.assertEqual(len(results), 2)

    def test_retrieve_returns_empty_for_empty_query(self) -> None:
        chunks = [build_chunk("a", "RAG evaluation faithfulness.")]

        self.assertEqual(BM25Retriever(chunks).retrieve("", top_k=3), [])

    def test_retrieve_returns_empty_for_non_positive_top_k(self) -> None:
        chunks = [build_chunk("a", "RAG evaluation faithfulness.")]

        self.assertEqual(BM25Retriever(chunks).retrieve("RAG", top_k=0), [])

    def test_repository_exposes_iterable_chunks_for_retriever(self) -> None:
        collection = InMemoryChunkCollection()
        collection.add_many([build_chunk("a", "RAG evaluation faithfulness.")])

        results = BM25Retriever(collection.iter_chunks()).retrieve("faithfulness", top_k=1)

        self.assertEqual(results[0].chunk_id, "a")


if __name__ == "__main__":
    unittest.main()
