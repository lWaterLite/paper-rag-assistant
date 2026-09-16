"""RAG 评测的确定性指标计算。"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from app.evaluation.models import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationMetrics,
    EvaluationSummary,
    MetricAggregate,
)
from app.generation.models import RagAnswer
from app.retrieval.context import PackedContext
from app.retrieval.models import RetrievedChunk


def calculate_case_metrics(
    case: EvaluationCase,
    *,
    retrieved_chunks: Iterable[RetrievedChunk],
    packed_context: PackedContext,
    answer: RagAnswer,
) -> EvaluationMetrics:
    """计算单样例的检索、上下文、引用与拒答自动指标。"""

    retrieved = tuple(retrieved_chunks)
    expected_ids, identity = _expected_evidence(case)
    retrieved_ids = _select_ids(retrieved, identity)
    context_ids = _select_ids(packed_context.used_chunks, identity)
    citation_ids = _select_citation_ids(answer, identity)

    retrieval_metrics = _rank_metrics(expected_ids, retrieved_ids)
    context_metrics = _set_metrics(expected_ids, context_ids)
    citation_coverage = _coverage(expected_ids, citation_ids)
    answered = answer.status != "abstained"
    return EvaluationMetrics(
        retrieval_hit_rate=retrieval_metrics["hit_rate"],
        retrieval_recall=retrieval_metrics["recall"],
        retrieval_reciprocal_rank=retrieval_metrics["reciprocal_rank"],
        retrieval_precision=retrieval_metrics["precision"],
        context_recall=context_metrics["recall"],
        context_precision=context_metrics["precision"],
        citation_coverage=citation_coverage,
        answer_has_citation=1.0 if answered and answer.citations else 0.0 if answered else None,
        abstention_correct=float(answer.status == "abstained")
        if case.expected_abstention
        else float(answer.status != "abstained"),
    )


def summarize_results(results: Iterable[EvaluationCaseResult]) -> EvaluationSummary:
    """按全体和问题类型计算宏平均指标，并统计失败根因。"""

    all_results = tuple(results)
    successful = tuple(result for result in all_results if result.status == "success")
    by_type: dict[str, list[EvaluationCaseResult]] = {}
    failure_counts: dict[str, int] = {}
    for result in all_results:
        by_type.setdefault(result.case.question_type, []).append(result)
        if result.failure is not None:
            failure_type = result.failure.failure_type
            failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1
    return EvaluationSummary(
        total_case_count=len(all_results),
        success_case_count=len(successful),
        error_case_count=len(all_results) - len(successful),
        metrics=_aggregate_metrics(successful),
        metrics_by_question_type={
            question_type: _aggregate_metrics(
                tuple(item for item in group if item.status == "success")
            )
            for question_type, group in sorted(by_type.items())
        },
        failure_counts=dict(sorted(failure_counts.items())),
    )


def _expected_evidence(case: EvaluationCase) -> tuple[set[str], str]:
    if case.expected_chunk_ids:
        return set(case.expected_chunk_ids), "chunk"
    if case.expected_doc_ids:
        return set(case.expected_doc_ids), "document"
    if case.expected_source_paths:
        return {_normalize_source_path(path) for path in case.expected_source_paths}, "source"
    return set(), "none"


def _select_ids(chunks: Iterable[RetrievedChunk], identity: str) -> tuple[str, ...]:
    if identity == "chunk":
        return tuple(chunk.chunk_id for chunk in chunks)
    if identity == "document":
        return tuple(chunk.doc_id for chunk in chunks)
    if identity == "source":
        return tuple(_normalize_source_path(chunk.source_path) for chunk in chunks)
    return ()


def _select_citation_ids(answer: RagAnswer, identity: str) -> tuple[str, ...]:
    if identity == "chunk":
        return tuple(citation.chunk_id for citation in answer.citations)
    if identity == "document":
        return tuple(citation.doc_id for citation in answer.citations)
    if identity == "source":
        return tuple(_normalize_source_path(citation.source_path) for citation in answer.citations)
    return ()


def _normalize_source_path(value: str) -> str:
    """将跨平台路径统一为可写入黄金集的 POSIX 形式。"""

    return Path(value).as_posix()


def _rank_metrics(expected_ids: set[str], actual_ids: tuple[str, ...]) -> dict[str, float | None]:
    if not expected_ids:
        return {"hit_rate": None, "recall": None, "reciprocal_rank": None, "precision": None}
    matched = [item for item in actual_ids if item in expected_ids]
    first_rank = next((rank for rank, item in enumerate(actual_ids, 1) if item in expected_ids), None)
    return {
        "hit_rate": float(bool(matched)),
        "recall": len(set(matched)) / len(expected_ids),
        "reciprocal_rank": 1.0 / first_rank if first_rank is not None else 0.0,
        "precision": len(matched) / len(actual_ids) if actual_ids else 0.0,
    }


def _set_metrics(expected_ids: set[str], actual_ids: tuple[str, ...]) -> dict[str, float | None]:
    if not expected_ids:
        return {"recall": None, "precision": None}
    actual_set = set(actual_ids)
    matches = expected_ids & actual_set
    return {
        "recall": len(matches) / len(expected_ids),
        "precision": len(matches) / len(actual_set) if actual_set else 0.0,
    }


def _coverage(expected_ids: set[str], actual_ids: tuple[str, ...]) -> float | None:
    if not expected_ids:
        return None
    return len(expected_ids & set(actual_ids)) / len(expected_ids)


def _aggregate_metrics(
    results: Iterable[EvaluationCaseResult],
) -> dict[str, MetricAggregate]:
    values_by_name: dict[str, list[float]] = {}
    for result in results:
        for name, value in result.metrics.as_mapping().items():
            if value is not None:
                values_by_name.setdefault(name, []).append(value)
    return {
        name: MetricAggregate(
            value=round(sum(values) / len(values), 6) if values else None,
            evaluated_case_count=len(values),
        )
        for name, values in sorted(values_by_name.items())
    }
