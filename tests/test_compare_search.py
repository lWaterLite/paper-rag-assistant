"""Compare search 测试。"""

from __future__ import annotations

import unittest

from app.api.handlers import handle_compare_search_request
from app.api.schemas import CompareSearchRequest
from app.core.models import RetrievedChunk
from app.retrieval.configs import RetrievalConfig
from app.retrieval.reporting import RetrievalReporter
from app.retrieval.retrievers import RetrieverRegistry
from app.retrieval.service import CompareSearchService


def build_result(
    chunk_id: str,
    *,
    rank: int = 1,
    retriever: str = "vector",
) -> RetrievedChunk:
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
    )


class StaticRetriever:
    """返回固定结果的测试检索器。"""

    def __init__(self, results: list[RetrievedChunk]) -> None:
        self._results = results

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        _ = query
        return list(self._results[:top_k])


def build_registry(**retrievers: StaticRetriever) -> RetrieverRegistry:
    """创建测试用检索器注册表。"""

    registry = RetrieverRegistry()
    for name, retriever in retrievers.items():
        registry.register(name, lambda retriever=retriever: retriever)
    return registry


def build_service(registry: RetrieverRegistry) -> CompareSearchService:
    """创建 compare search 服务。"""

    return CompareSearchService(
        registry=registry,
        config=RetrievalConfig(strategy="vector", top_k=2),
        reporter=RetrievalReporter.disabled(),
    )


class CompareSearchTest(unittest.TestCase):
    """验证多策略检索比较流程。"""

    def test_compare_search_returns_results_and_overlaps(self) -> None:
        service = build_service(
            build_registry(
                vector=StaticRetriever(
                    [
                        build_result("shared", rank=1, retriever="vector"),
                        build_result("vector_only", rank=2, retriever="vector"),
                    ]
                ),
                bm25=StaticRetriever(
                    [
                        build_result("bm25_only", rank=1, retriever="bm25"),
                        build_result("shared", rank=2, retriever="bm25"),
                    ]
                ),
            )
        )

        result = service.compare(
            "faithfulness",
            retrievers=["vector", "bm25"],
            top_k=2,
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.retrievers, ("vector", "bm25"))
        self.assertEqual([item.status for item in result.strategy_results], ["success", "success"])
        self.assertEqual(result.overlaps[0].chunk_id, "shared")
        self.assertEqual(
            result.overlaps[0].ranks_by_retriever,
            {"vector": 1, "bm25": 2},
        )
        self.assertEqual(result.trace.final_status, "success")

    def test_compare_search_keeps_successful_results_when_one_strategy_fails(self) -> None:
        service = build_service(
            build_registry(
                vector=StaticRetriever(
                    [build_result("chunk_a", rank=1, retriever="vector")]
                ),
            )
        )

        result = service.compare(
            "RAG",
            retrievers=["vector", "missing"],
            top_k=1,
        )

        self.assertEqual(result.status, "partial_error")
        self.assertEqual(result.strategy_results[0].status, "success")
        self.assertEqual(result.strategy_results[1].status, "error")
        self.assertEqual(result.strategy_results[1].error_code, "INVALID_CONFIG")
        self.assertEqual(result.trace.final_status, "success")

    def test_compare_search_handler_maps_domain_result_to_api_response(self) -> None:
        service = build_service(
            build_registry(
                vector=StaticRetriever(
                    [build_result("chunk_a", rank=1, retriever="vector")]
                ),
                bm25=StaticRetriever(
                    [build_result("chunk_a", rank=1, retriever="bm25")]
                ),
            )
        )

        response = handle_compare_search_request(
            CompareSearchRequest(
                query="  citation  ",
                retrievers=["vector", "bm25"],
                top_k=1,
                debug_trace=True,
            ),
            service,
        )

        self.assertEqual(response.query, "citation")
        self.assertEqual(response.status, "success")
        self.assertEqual(response.strategy_results[0].retriever, "vector")
        self.assertEqual(response.strategy_results[0].results[0].chunk_id, "chunk_a")
        self.assertEqual(response.overlaps[0].retrievers, ["vector", "bm25"])
        self.assertIsNotNone(response.trace)
        self.assertIsNotNone(response.strategy_results[0].trace)


if __name__ == "__main__":
    unittest.main()
