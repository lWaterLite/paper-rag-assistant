"""Hybrid Retriever 与 RRF 融合测试。"""

from __future__ import annotations

import unittest
from collections.abc import Sequence
from dataclasses import replace

from app.retrieval.models import RetrievedChunk
from app.retrieval.configuration import HybridRetrievalConfig
from app.retrieval.retrievers.fusion.base import RankedResultSet
from app.retrieval.retrievers.fusion.rrf import ReciprocalRankFusion
from app.retrieval.retrievers.hybrid import (
    HybridRetrievalSource,
    HybridRetriever,
)


def build_result(
    chunk_id: str,
    *,
    score: float,
    rank: int,
    retriever: str,
) -> RetrievedChunk:
    """创建测试用检索结果。"""

    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        content_hash=f"hash_{chunk_id}",
        version_id=f"version_{chunk_id}",
        text=f"chunk {chunk_id}",
        score=score,
        rank=rank,
        retriever=retriever,
        source_path=f"{chunk_id}.md",
        chunk_index=0,
    )


class RecordingRetriever:
    """记录调用参数并返回预设候选的测试检索器。"""

    def __init__(self, results: Sequence[RetrievedChunk]) -> None:
        self._results = list(results)
        self.requested_top_k: list[int] = []

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        _ = query
        self.requested_top_k.append(top_k)
        return self._results[:top_k]


class ReciprocalRankFusionTest(unittest.TestCase):
    """验证 RRF 排名融合。"""

    def test_common_chunk_receives_signals_from_both_sources(self) -> None:
        vector_results = [
            build_result("a", score=0.95, rank=1, retriever="vector"),
            build_result("b", score=0.80, rank=2, retriever="vector"),
        ]
        bm25_results = [
            build_result("b", score=8.5, rank=1, retriever="bm25"),
            build_result("c", score=7.0, rank=2, retriever="bm25"),
        ]

        hits = ReciprocalRankFusion(rank_constant=60).fuse(
            [
                RankedResultSet("vector", 1.0, vector_results),
                RankedResultSet("bm25", 1.0, bm25_results),
            ],
            limit=3,
        )

        self.assertEqual([hit.chunk.chunk_id for hit in hits], ["b", "a", "c"])
        self.assertEqual(
            [signal.retriever for signal in hits[0].signals],
            ["vector", "bm25"],
        )

    def test_source_weight_affects_fused_order(self) -> None:
        vector_result = build_result(
            "vector_only", score=0.9, rank=1, retriever="vector"
        )
        bm25_result = build_result(
            "bm25_only", score=100.0, rank=1, retriever="bm25"
        )

        hits = ReciprocalRankFusion(rank_constant=60).fuse(
            [
                RankedResultSet("vector", 2.0, [vector_result]),
                RankedResultSet("bm25", 1.0, [bm25_result]),
            ],
            limit=2,
        )

        self.assertEqual(hits[0].chunk.chunk_id, "vector_only")


class HybridRetrieverTest(unittest.TestCase):
    """验证 Hybrid Retriever 的召回编排。"""

    def test_retrieve_expands_candidates_and_returns_hybrid_results(self) -> None:
        vector = RecordingRetriever(
            [
                build_result("a", score=0.9, rank=1, retriever="vector"),
                build_result("b", score=0.8, rank=2, retriever="vector"),
            ]
        )
        bm25 = RecordingRetriever(
            [
                build_result("b", score=9.0, rank=1, retriever="bm25"),
                build_result("c", score=8.0, rank=2, retriever="bm25"),
            ]
        )
        retriever = HybridRetriever(
            sources=(
                HybridRetrievalSource("vector", vector, 1.0),
                HybridRetrievalSource("bm25", bm25, 1.0),
            ),
            fusion_strategy=ReciprocalRankFusion(rank_constant=60),
            config=HybridRetrievalConfig(candidate_multiplier=3),
        )

        results = retriever.retrieve("query", top_k=2)

        self.assertEqual(vector.requested_top_k, [6])
        self.assertEqual(bm25.requested_top_k, [6])
        self.assertEqual([result.chunk_id for result in results], ["b", "a"])
        self.assertEqual([result.rank for result in results], [1, 2])
        self.assertTrue(all(result.retriever == "hybrid" for result in results))
        self.assertEqual(
            [signal.retriever for signal in results[0].retrieval_signals],
            ["vector", "bm25"],
        )

    def test_retrieve_does_not_mutate_source_result(self) -> None:
        source_result = build_result("a", score=0.9, rank=1, retriever="vector")
        vector = RecordingRetriever([source_result])
        bm25 = RecordingRetriever(
            [replace(source_result, score=9.0, retriever="bm25")]
        )
        retriever = HybridRetriever(
            sources=(
                HybridRetrievalSource("vector", vector, 1.0),
                HybridRetrievalSource("bm25", bm25, 1.0),
            ),
            fusion_strategy=ReciprocalRankFusion(),
            config=HybridRetrievalConfig(),
        )

        retriever.retrieve("query", top_k=1)

        self.assertEqual(source_result.retriever, "vector")
        self.assertEqual(source_result.retrieval_signals, ())


if __name__ == "__main__":
    unittest.main()
