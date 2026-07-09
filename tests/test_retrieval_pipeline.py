"""RetrievalPipeline 测试。"""

from __future__ import annotations

import unittest

from app.core.models import RetrievedChunk
from app.retrieval.configs import RetrievalConfig
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.retrievers import RetrieverRegistry


def build_result(chunk_id: str, *, rank: int = 1, retriever: str = "vector") -> RetrievedChunk:
    """构造测试检索结果。"""

    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        content_hash=f"hash_{chunk_id}",
        version_id=f"v_{chunk_id}",
        text=f"{chunk_id} 的测试文本",
        score=1.0 / rank,
        rank=rank,
        retriever=retriever,
        source_path=f"{chunk_id}.md",
        chunk_index=rank - 1,
        title=f"文档 {chunk_id}",
        section="测试章节",
        metadata={"source": "test"},
    )


class StaticRetriever:
    """返回固定结果的测试检索器。"""

    def __init__(self, results: list[RetrievedChunk]) -> None:
        self._results = results

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """返回固定结果，让测试只关注 pipeline 后处理行为。"""

        _ = query, top_k
        return list(self._results)


def build_registry(**retrievers: StaticRetriever) -> RetrieverRegistry:
    """把测试检索器注册为惰性 provider。"""

    registry = RetrieverRegistry()
    for name, retriever in retrievers.items():
        registry.register(name, lambda retriever=retriever: retriever)
    return registry


class RetrievalPipelineTest(unittest.TestCase):
    """验证 retrieval pipeline 的阶段编排。"""

    def test_pipeline_deduplicates_and_limits_results(self) -> None:
        pipeline = RetrievalPipeline(
            registry=build_registry(
                vector=StaticRetriever(
                    [
                        build_result("chunk_a", rank=1),
                        build_result("chunk_a", rank=2),
                        build_result("chunk_b", rank=3),
                        build_result("chunk_c", rank=4),
                    ]
                ),
            ),
            config=RetrievalConfig(strategy="vector", top_k=2, deduplicate_by_chunk_id=True),
        )

        result = pipeline.search("RAG citation")

        self.assertEqual([chunk.chunk_id for chunk in result.results], ["chunk_a", "chunk_b"])
        self.assertEqual([chunk.rank for chunk in result.results], [1, 2])
        self.assertEqual(result.trace.final_status, "success")

    def test_pipeline_can_keep_duplicate_chunks_when_configured(self) -> None:
        pipeline = RetrievalPipeline(
            registry=build_registry(
                vector=StaticRetriever(
                    [
                        build_result("chunk_a", rank=1),
                        build_result("chunk_a", rank=2),
                    ]
                ),
            ),
            config=RetrievalConfig(strategy="vector", top_k=2, deduplicate_by_chunk_id=False),
        )

        result = pipeline.search("RAG citation")

        self.assertEqual([chunk.chunk_id for chunk in result.results], ["chunk_a", "chunk_a"])
        self.assertEqual([chunk.rank for chunk in result.results], [1, 2])


if __name__ == "__main__":
    unittest.main()
