"""评测自动指标计算测试。"""

from __future__ import annotations

import unittest
from dataclasses import replace

from app.evaluation.metrics import calculate_case_metrics, summarize_results
from app.evaluation.models import EvaluationCase, EvaluationCaseResult
from app.generation.models import Citation, RagAnswer
from app.retrieval.context.packer import (
    ContextCitation,
    ContextTokenUsage,
    PackedContext,
)
from app.retrieval.models import RetrievedChunk


class EvaluationMetricsTest(unittest.TestCase):
    """验证检索、上下文、引用和拒答指标的明确语义。"""

    def test_calculates_metrics_against_expected_chunk_ids(self) -> None:
        case = EvaluationCase(
            case_id="fact-001",
            question="主要结论是什么？",
            question_type="fact",
            expected_chunk_ids=("chunk-expected",),
        )
        expected_chunk = _chunk("chunk-expected", "paper-a", rank=2)
        distractor_chunk = _chunk("chunk-other", "paper-b", rank=1)
        context = _packed_context([expected_chunk])
        answer = _answer([expected_chunk])

        metrics = calculate_case_metrics(
            case,
            retrieved_chunks=[distractor_chunk, expected_chunk],
            packed_context=context,
            answer=answer,
        )

        self.assertEqual(metrics.retrieval_hit_rate, 1.0)
        self.assertEqual(metrics.retrieval_recall, 1.0)
        self.assertEqual(metrics.retrieval_reciprocal_rank, 0.5)
        self.assertEqual(metrics.retrieval_precision, 0.5)
        self.assertEqual(metrics.context_recall, 1.0)
        self.assertEqual(metrics.context_precision, 1.0)
        self.assertEqual(metrics.citation_coverage, 1.0)
        self.assertEqual(metrics.answer_has_citation, 1.0)
        self.assertEqual(metrics.abstention_correct, 1.0)

    def test_calculates_abstention_accuracy_without_fabricating_evidence_metrics(self) -> None:
        case = EvaluationCase(
            case_id="abstain-001",
            question="资料中不存在的内容是什么？",
            question_type="abstention",
            expected_abstention=True,
        )
        answer = RagAnswer(
            answer="资料中没有足够证据回答该问题。",
            citations=[],
            retrieved_chunks=[],
            trace_id="trace_test",
            latency_ms=1.0,
            status="abstained",
            abstention_reason="no_supported_evidence",
        )

        metrics = calculate_case_metrics(
            case,
            retrieved_chunks=[],
            packed_context=_packed_context([]),
            answer=answer,
        )

        self.assertIsNone(metrics.retrieval_hit_rate)
        self.assertIsNone(metrics.context_recall)
        self.assertIsNone(metrics.citation_coverage)
        self.assertIsNone(metrics.answer_has_citation)
        self.assertEqual(metrics.abstention_correct, 1.0)

    def test_matches_expected_source_path_across_path_separators(self) -> None:
        case = EvaluationCase(
            case_id="source-001",
            question="来源路径是否可跨平台匹配？",
            question_type="fact",
            expected_source_paths=("data/raw/papers/paper-a.md",),
        )
        chunk = _chunk("chunk-a", "paper-a", rank=1)
        chunk = replace(chunk, source_path="data\\raw\\papers\\paper-a.md")

        metrics = calculate_case_metrics(
            case,
            retrieved_chunks=[chunk],
            packed_context=_packed_context([chunk]),
            answer=_answer([chunk]),
        )

        self.assertEqual(metrics.retrieval_hit_rate, 1.0)
        self.assertEqual(metrics.context_recall, 1.0)
        self.assertEqual(metrics.citation_coverage, 1.0)

    def test_summary_uses_only_successful_results_for_metric_average(self) -> None:
        case = EvaluationCase(
            case_id="fact-001",
            question="主要结论是什么？",
            question_type="fact",
            expected_doc_ids=("paper-a",),
        )
        metrics = calculate_case_metrics(
            case,
            retrieved_chunks=[_chunk("chunk-a", "paper-a", rank=1)],
            packed_context=_packed_context([_chunk("chunk-a", "paper-a", rank=1)]),
            answer=_answer([_chunk("chunk-a", "paper-a", rank=1)]),
        )
        summary = summarize_results(
            [EvaluationCaseResult(case=case, status="success", metrics=metrics)]
        )

        self.assertEqual(summary.total_case_count, 1)
        self.assertEqual(summary.success_case_count, 1)
        self.assertEqual(summary.metrics["retrieval_hit_rate"].value, 1.0)


def _chunk(chunk_id: str, doc_id: str, *, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content_hash=f"hash-{chunk_id}",
        version_id="v1",
        text=f"{chunk_id} 的资料正文",
        score=1.0 / rank,
        rank=rank,
        retriever="vector",
        source_path=f"data/raw/{doc_id}.md",
        chunk_index=rank - 1,
        title=doc_id,
    )


def _packed_context(chunks: list[RetrievedChunk]) -> PackedContext:
    citations = [
        ContextCitation(
            citation_id=f"C{position}",
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            version_id=chunk.version_id,
            title=chunk.title,
            source_path=chunk.source_path,
            snippet=chunk.text,
        )
        for position, chunk in enumerate(chunks, 1)
    ]
    return PackedContext(
        context_text="\n".join(chunk.text for chunk in chunks),
        citations=citations,
        used_chunks=chunks,
        dropped_chunks=[],
        segments=[],
        token_usage=ContextTokenUsage(
            estimator="regex",
            question_tokens=1,
            reserved_prompt_tokens=1,
            reserved_output_tokens=1,
            safety_margin_tokens=1,
            available_context_tokens=100,
            used_context_tokens=len(chunks),
        ),
    )


def _answer(chunks: list[RetrievedChunk]) -> RagAnswer:
    citations = [
        Citation(
            citation_id=f"C{position}",
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            version_id=chunk.version_id,
            title=chunk.title,
            source_path=chunk.source_path,
            snippet=chunk.text,
        )
        for position, chunk in enumerate(chunks, 1)
    ]
    return RagAnswer(
        answer="测试回答",
        citations=citations,
        retrieved_chunks=chunks,
        trace_id="trace_test",
        latency_ms=1.0,
    )


if __name__ == "__main__":
    unittest.main()
