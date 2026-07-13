"""Reranker、候选上限与失败降级测试。"""

from __future__ import annotations

import unittest

from app.core.errors import AppError, ErrorCode
from app.core.models import RetrievedChunk
from app.retrieval.configs import RetrievalConfig
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.reporting import RetrievalReporter
from app.retrieval.rerankers import (
    LexicalReranker,
    RerankingConfig,
    build_default_reranker_registry,
)
from app.retrieval.retrievers import RetrieverRegistry
from app.retrieval.tokenizers import RegexTokenizer


def build_chunk(chunk_id: str, text: str, *, rank: int) -> RetrievedChunk:
    """构造可排序的检索候选。"""

    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        content_hash=f"hash_{chunk_id}",
        version_id="v1",
        text=text,
        score=1.0 / rank,
        rank=rank,
        retriever="vector",
        source_path=f"{chunk_id}.md",
        chunk_index=rank - 1,
    )


class StaticRetriever:
    """按调用方候选上限返回固定检索结果。"""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        _ = query
        return list(self._chunks[:top_k])


class FailingReranker:
    """用于验证 rerank 失败策略。"""

    @property
    def name(self) -> str:
        return "failing"

    def rerank(self, query, candidates, *, limit):
        _ = query, candidates, limit
        raise RuntimeError("rerank 服务不可用")


def build_registry(chunks: list[RetrievedChunk]) -> RetrieverRegistry:
    """创建测试用 retriever registry。"""

    registry = RetrieverRegistry()
    registry.register("vector", lambda: StaticRetriever(chunks))
    return registry


class RerankingTest(unittest.TestCase):
    """验证 rerank 是独立后处理阶段，而不是 retriever 内部逻辑。"""

    def test_lexical_reranker_promotes_chunk_with_better_query_coverage(self) -> None:
        reranker = LexicalReranker(RegexTokenizer(), batch_size=2)
        reranked = reranker.rerank(
            "rerank latency",
            [
                build_chunk("generic", "rerank improves ordering", rank=1),
                build_chunk("direct", "rerank adds latency during retrieval", rank=2),
            ],
            limit=2,
        )

        self.assertEqual(reranked[0].chunk.chunk_id, "direct")
        self.assertGreater(reranked[0].score, reranked[1].score)

    def test_registry_creates_configured_lexical_reranker(self) -> None:
        registry = build_default_reranker_registry(RegexTokenizer())

        reranker = registry.create(
            RerankingConfig(enabled=True, strategy="lexical", batch_size=2)
        )

        self.assertEqual(reranker.name, "lexical")

    def test_pipeline_uses_candidate_limit_before_final_top_k(self) -> None:
        pipeline = RetrievalPipeline(
            registry=build_registry(
                [
                    build_chunk("generic", "rerank improves ordering", rank=1),
                    build_chunk("direct", "rerank adds latency during retrieval", rank=2),
                    build_chunk("other", "BM25 retrieves keywords", rank=3),
                ]
            ),
            config=RetrievalConfig(strategy="vector", top_k=1),
            reranking_config=RerankingConfig(
                enabled=True,
                candidate_limit=3,
            ),
            reranker=LexicalReranker(RegexTokenizer(), batch_size=2),
            reporter=RetrievalReporter.disabled(),
        )

        result = pipeline.search("rerank latency")

        self.assertEqual(result.candidate_limit, 3)
        self.assertEqual([chunk.chunk_id for chunk in result.results], ["direct"])
        self.assertEqual(result.results[0].rerank_signal.reranker, "lexical")
        self.assertIn("RerankStage", [stage.stage for stage in result.trace.stages])

    def test_fail_open_keeps_original_order_and_records_degraded_stage(self) -> None:
        pipeline = RetrievalPipeline(
            registry=build_registry(
                [
                    build_chunk("first", "first text", rank=1),
                    build_chunk("second", "second text", rank=2),
                ]
            ),
            config=RetrievalConfig(strategy="vector", top_k=2),
            reranking_config=RerankingConfig(
                enabled=True,
                candidate_limit=2,
                failure_mode="fail_open",
            ),
            reranker=FailingReranker(),
            reporter=RetrievalReporter.disabled(),
        )

        result = pipeline.search("RAG")

        self.assertEqual([chunk.chunk_id for chunk in result.results], ["first", "second"])
        rerank_stage = next(
            stage for stage in result.trace.stages if stage.stage == "RerankStage"
        )
        self.assertTrue(rerank_stage.detail["degraded"])
        self.assertEqual(rerank_stage.detail["reranker"], "failing")

    def test_fail_closed_raises_rerank_error(self) -> None:
        pipeline = RetrievalPipeline(
            registry=build_registry([build_chunk("first", "first text", rank=1)]),
            config=RetrievalConfig(strategy="vector", top_k=1),
            reranking_config=RerankingConfig(
                enabled=True,
                candidate_limit=1,
                failure_mode="fail_closed",
            ),
            reranker=FailingReranker(),
            reporter=RetrievalReporter.disabled(),
        )

        with self.assertRaises(AppError) as context:
            pipeline.search("RAG")

        self.assertEqual(context.exception.code, ErrorCode.RERANK_FAILED)


if __name__ == "__main__":
    unittest.main()
