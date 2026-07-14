"""SearchService 测试。"""

from __future__ import annotations

import unittest

from app.api.handlers import handle_search_request
from app.api.contracts.retrieval import SearchRequest
from app.core.errors import AppError, ErrorCode
from app.core.models import RetrievedChunk
from app.retrieval.configuration import RetrievalConfig
from app.retrieval.retrievers import RetrieverRegistry
from app.retrieval.reporting import RetrievalReporter
from app.retrieval.rerankers import RerankingConfig
from app.retrieval.services.search import SearchService


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

    def __init__(self, name: str, results: list[RetrievedChunk]) -> None:
        self._name = name
        self._results = results

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        _ = query
        return [
            result
            for result in self._results[:top_k]
        ]


def build_registry(**retrievers: StaticRetriever) -> RetrieverRegistry:
    """创建包含指定测试检索器的注册表。"""

    registry = RetrieverRegistry()
    for name, retriever in retrievers.items():
        registry.register(name, lambda retriever=retriever: retriever)
    return registry


class SearchServiceTest(unittest.TestCase):
    """验证 search 服务编排。"""

    def test_search_uses_default_retriever_and_deduplicates_chunks(self) -> None:
        service = SearchService(
            registry=build_registry(
                vector=StaticRetriever(
                    "vector",
                    [
                        build_result("chunk_a", rank=1, retriever="vector"),
                        build_result("chunk_a", rank=2, retriever="vector"),
                        build_result("chunk_b", rank=3, retriever="vector"),
                    ],
                ),
            ),
            config=RetrievalConfig(strategy="vector", top_k=3, deduplicate_by_chunk_id=True),
            reranking_config=RerankingConfig(enabled=False),
            reranker=None,
            reporter=RetrievalReporter.disabled(),
        )

        result = service.search("  RAG citation  ")

        self.assertEqual(result.query, "RAG citation")
        self.assertEqual(result.retriever, "vector")
        self.assertEqual([chunk.chunk_id for chunk in result.results], ["chunk_a", "chunk_b"])
        self.assertEqual([chunk.rank for chunk in result.results], [1, 2])
        self.assertEqual(result.trace.final_status, "success")

    def test_search_allows_request_level_retriever_override(self) -> None:
        service = SearchService(
            registry=build_registry(
                vector=StaticRetriever(
                    "vector",
                    [build_result("vector_chunk", retriever="vector")],
                ),
                bm25=StaticRetriever(
                    "bm25",
                    [build_result("bm25_chunk", retriever="bm25")],
                ),
            ),
            config=RetrievalConfig(strategy="vector", top_k=1),
            reranking_config=RerankingConfig(enabled=False),
            reranker=None,
            reporter=RetrievalReporter.disabled(),
        )

        result = service.search("faithfulness", retriever="bm25")

        self.assertEqual(result.retriever, "bm25")
        self.assertEqual(result.results[0].chunk_id, "bm25_chunk")

    def test_search_rejects_unsupported_retriever(self) -> None:
        service = SearchService(
            registry=build_registry(vector=StaticRetriever("vector", [])),
            config=RetrievalConfig(strategy="vector", top_k=1),
            reranking_config=RerankingConfig(enabled=False),
            reranker=None,
            reporter=RetrievalReporter.disabled(),
        )

        with self.assertRaises(AppError) as context:
            service.search("RAG", retriever="hybrid")

        self.assertEqual(context.exception.code, ErrorCode.INVALID_CONFIG)
        self.assertIn("hybrid", context.exception.message)

    def test_search_handler_maps_domain_result_to_api_response(self) -> None:
        service = SearchService(
            registry=build_registry(
                bm25=StaticRetriever(
                    "bm25",
                    [build_result("chunk_a", retriever="bm25")],
                )
            ),
            config=RetrievalConfig(strategy="bm25", top_k=1),
            reranking_config=RerankingConfig(enabled=False),
            reranker=None,
            reporter=RetrievalReporter.disabled(),
        )

        response = handle_search_request(
            SearchRequest(query="faithfulness", retriever="bm25", debug_trace=True),
            service,
        )

        self.assertEqual(response.query, "faithfulness")
        self.assertEqual(response.retriever, "bm25")
        self.assertEqual(response.results[0].chunk_id, "chunk_a")
        self.assertIsNotNone(response.trace)
        self.assertEqual(response.trace.final_status, "success")


if __name__ == "__main__":
    unittest.main()
