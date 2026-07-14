"""BM25Retriever 测试。"""

from __future__ import annotations

import unittest
from collections.abc import Iterable, Sequence

from app.ingest.chunking.models import DocumentChunk
from app.ingest.chunking.collection import InMemoryChunkCollection
from app.retrieval.configuration import BM25Config
from app.retrieval.retrievers import BM25Index, BM25Retriever
from app.retrieval.tokenizers import RegexTokenizer, Tokenizer


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


def build_retriever(
    chunks: Iterable[DocumentChunk],
    tokenizer: Tokenizer | None = None,
) -> BM25Retriever:
    index = BM25Index.from_chunks(
        chunks,
        config=BM25Config(),
        tokenizer=tokenizer or RegexTokenizer(),
    )
    return BM25Retriever(index)


class BM25RetrieverTest(unittest.TestCase):
    """验证 BM25 关键词检索。"""

    def test_retrieve_returns_ranked_retrieved_chunks(self) -> None:
        chunks = [
            build_chunk("a", "RAG evaluation includes faithfulness.", "Evaluation"),
            build_chunk("b", "Vector databases store embedding vectors.", "Indexing"),
        ]
        results = build_retriever(chunks).retrieve(
            "faithfulness evaluation", top_k=2
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "a")
        self.assertEqual(results[0].retriever, "bm25")
        self.assertEqual(results[0].rank, 1)
        self.assertGreater(results[0].score, 0)

    def test_retrieve_preserves_chunk_metadata(self) -> None:
        chunk = build_chunk("a", "RAG evaluation includes faithfulness.", "Evaluation")
        result = build_retriever([chunk]).retrieve("faithfulness", top_k=1)[0]

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
        results = build_retriever(chunks).retrieve("RAG evaluation", top_k=2)

        self.assertEqual(len(results), 2)

    def test_retrieve_returns_empty_for_empty_query(self) -> None:
        chunks = [build_chunk("a", "RAG evaluation faithfulness.")]

        self.assertEqual(build_retriever(chunks).retrieve("", top_k=3), [])

    def test_retrieve_returns_empty_for_non_positive_top_k(self) -> None:
        chunks = [build_chunk("a", "RAG evaluation faithfulness.")]

        self.assertEqual(build_retriever(chunks).retrieve("RAG", top_k=0), [])

    def test_repository_exposes_iterable_chunks_for_retriever(self) -> None:
        collection = InMemoryChunkCollection()
        collection.add_many([build_chunk("a", "RAG evaluation faithfulness.")])

        results = build_retriever(collection.iter_chunks()).retrieve(
            "faithfulness", top_k=1
        )

        self.assertEqual(results[0].chunk_id, "a")

    def test_index_and_query_share_injected_tokenizer(self) -> None:
        class RecordingTokenizer:
            def __init__(self) -> None:
                self.inputs: list[str] = []

            def tokenize(self, text: str) -> Sequence[str]:
                self.inputs.append(text)
                return text.lower().split()

        tokenizer = RecordingTokenizer()
        retriever = build_retriever(
            [build_chunk("a", "shared tokenizer")],
            tokenizer=tokenizer,
        )

        results = retriever.retrieve("shared", top_k=1)

        self.assertEqual(results[0].chunk_id, "a")
        self.assertEqual(tokenizer.inputs, ["shared tokenizer", "shared"])


if __name__ == "__main__":
    unittest.main()
