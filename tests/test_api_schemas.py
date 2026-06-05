"""API schema 设计测试。"""

from __future__ import annotations

import time
import unittest

from pydantic import ValidationError

from app.api.routes import api_contract, planned_routes
from app.api.schemas import (
    AskRequest,
    SearchRequest,
    rag_answer_to_response,
    retrieved_chunk_to_response,
    trace_to_response,
)
from app.core.models import Citation, RagAnswer, RagTrace, RetrievedChunk


def build_retrieved_chunk() -> RetrievedChunk:
    """构造一个检索结果，用于测试 API 映射。"""

    return RetrievedChunk(
        chunk_id="chunk_1",
        doc_id="doc_1",
        content_hash="hash_1",
        version_id="v_1",
        text="RAG 需要把回答约束在检索到的上下文中。",
        score=0.87,
        rank=1,
        retriever="bm25",
        source_path="docs/rag.md",
        chunk_index=3,
        title="RAG 入门",
        section="生成阶段",
        metadata={"language": "zh"},
    )


def build_answer() -> RagAnswer:
    """构造一个 RAG 回答，用于测试 /ask 响应。"""

    chunk = build_retrieved_chunk()
    citation = Citation(
        citation_id="C1",
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        version_id=chunk.version_id,
        title=chunk.title,
        source_path=chunk.source_path,
        snippet=chunk.text,
        section=chunk.section,
    )
    return RagAnswer(
        answer="RAG 会先检索相关上下文，再基于上下文生成回答。[C1]",
        citations=[citation],
        retrieved_chunks=[chunk],
        trace_id="trace_123",
        latency_ms=12.5,
    )


class ApiSchemaTest(unittest.TestCase):
    """API 请求与响应 schema 的自检。"""

    def test_ask_request_validates_question_and_top_k(self) -> None:
        request = AskRequest(question="  什么是 RAG？  ", top_k=3)

        self.assertEqual(request.question, "什么是 RAG？")
        self.assertEqual(request.top_k, 3)

        with self.assertRaises(ValidationError):
            AskRequest(question="   ")

        with self.assertRaises(ValidationError):
            AskRequest(question="什么是 RAG？", top_k=0)

    def test_search_request_validates_query_and_retriever(self) -> None:
        request = SearchRequest(query="  embedding  ", retriever="hybrid")

        self.assertEqual(request.query, "embedding")
        self.assertEqual(request.retriever, "hybrid")

        with self.assertRaises(ValidationError):
            SearchRequest(query="")

        with self.assertRaises(ValidationError):
            SearchRequest(query="RAG", retriever="unknown")

    def test_retrieved_chunk_response_keeps_source_and_score(self) -> None:
        response = retrieved_chunk_to_response(build_retrieved_chunk())

        self.assertEqual(response.chunk_id, "chunk_1")
        self.assertEqual(response.score, 0.87)
        self.assertEqual(response.rank, 1)
        self.assertEqual(response.source_path, "docs/rag.md")
        self.assertEqual(response.metadata["language"], "zh")

    def test_ask_response_hides_retrieved_chunks_by_default(self) -> None:
        response = rag_answer_to_response(build_answer())

        self.assertEqual(response.trace_id, "trace_123")
        self.assertEqual(response.citations[0].citation_id, "C1")
        self.assertEqual(response.retrieved_chunks, [])

    def test_ask_response_can_include_retrieved_chunks(self) -> None:
        response = rag_answer_to_response(build_answer(), include_retrieved_chunks=True)

        self.assertEqual(len(response.retrieved_chunks), 1)
        self.assertEqual(response.retrieved_chunks[0].retriever, "bm25")

    def test_trace_response_exposes_final_status_and_stages(self) -> None:
        trace = RagTrace(trace_id="trace_failed")
        trace.record_stage(
            "retrieval",
            "error",
            time.perf_counter(),
            {"error_code": "RETRIEVAL_FAILED"},
        )
        trace.mark_failed("retrieval", "检索失败")

        response = trace_to_response(trace)

        self.assertEqual(response.trace_id, "trace_failed")
        self.assertEqual(response.final_status, "error")
        self.assertEqual(response.failure_type, "retrieval")
        self.assertEqual(response.stages[0].detail["error_code"], "RETRIEVAL_FAILED")

    def test_routes_expose_request_and_response_models(self) -> None:
        routes = {(route["method"], route["path"]): route for route in planned_routes()}

        self.assertEqual(routes[("POST", "/ask")]["request_model"], "AskRequest")
        self.assertEqual(routes[("POST", "/ask")]["response_model"], "AskResponse")
        self.assertEqual(routes[("POST", "/search")]["request_model"], "SearchRequest")
        self.assertEqual(routes[("POST", "/search")]["response_model"], "SearchResponse")
        self.assertEqual(routes[("GET", "/health")]["response_model"], "HealthResponse")

    def test_api_contract_declares_error_response(self) -> None:
        contract = api_contract()

        self.assertEqual(contract["error_response"]["response_model"], "ErrorResponse")


if __name__ == "__main__":
    unittest.main()
