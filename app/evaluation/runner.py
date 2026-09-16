"""复用生产 RAG Pipeline 的离线评测运行器。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from app.core.errors import AppError
from app.evaluation.configuration import EvaluationConfig, ExperimentConfig
from app.evaluation.datasets import EvaluationDatasetRepository
from app.evaluation.metrics import calculate_case_metrics, summarize_results
from app.evaluation.models import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationChunkObservation,
    EvaluationFailure,
    EvaluationIndexSnapshot,
    EvaluationRun,
    EvaluationRunResult,
)
from app.evaluation.reporting import EvaluationReportWriter
from app.evaluation.repositories import EvaluationResultRepository
from app.indexing.pipeline.types import RagIndex
from app.pipeline import RagPipelineExecutionResult


class EvaluationRagService(Protocol):
    """评测所需的最小 RAG 执行能力。"""

    def execute(
        self,
        question: str,
        *,
        top_k: int | None = None,
        retriever: str | None = None,
    ) -> RagPipelineExecutionResult:
        """执行完整 RAG 请求并返回阶段产物。"""


@dataclass(frozen=True, slots=True)
class EvaluationTarget:
    """一个实验 profile 与已组装 RAG 服务之间的绑定。"""

    experiment: ExperimentConfig
    service: EvaluationRagService


@dataclass(slots=True)
class EvaluationRunner:
    """执行单个实验 profile，并写入可复现的样例级结果。"""

    config: EvaluationConfig
    dataset_repository: EvaluationDatasetRepository
    result_repository: EvaluationResultRepository
    report_writer: EvaluationReportWriter

    def load_cases(self) -> tuple[EvaluationCase, ...]:
        """读取完整黄金评测集，并在配置要求时限制开发样例数。"""

        cases = self.dataset_repository.load(self.config.dataset_path)
        if self.config.max_cases is None:
            return cases
        return cases[: self.config.max_cases]

    def run(
        self,
        *,
        target: EvaluationTarget,
        index: RagIndex,
        cases: Sequence[EvaluationCase],
    ) -> EvaluationRunResult:
        """在固定索引、数据集和 profile 下运行一次完整实验。"""

        started_at = datetime.now(UTC).isoformat()
        run_id = _build_run_id(target.experiment.name)
        run_dir = self.config.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        results = tuple(
            self._run_case(case=case, target=target) for case in cases
        )
        run = EvaluationRun(
            run_id=run_id,
            dataset_id=self.config.dataset_id,
            dataset_version=self.config.dataset_version,
            experiment_name=target.experiment.name,
            experiment_snapshot={
                "retriever": target.experiment.retriever,
                "top_k": target.experiment.top_k,
                "reranking_enabled": target.experiment.reranking_enabled,
            },
            index_snapshot=_build_index_snapshot(index),
            case_results=results,
            summary=summarize_results(results),
            started_at=started_at,
        )
        artifacts = self.result_repository.write(run, run_dir)
        report_path = self.report_writer.write(run, Path(artifacts.report_path))
        return EvaluationRunResult(
            run=run,
            artifacts=replace(artifacts, report_path=report_path.as_posix()),
        )

    def _run_case(
        self,
        *,
        case: EvaluationCase,
        target: EvaluationTarget,
    ) -> EvaluationCaseResult:
        try:
            execution = target.service.execute(
                case.question,
                top_k=target.experiment.top_k,
                retriever=target.experiment.retriever,
            )
        except (AppError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return EvaluationCaseResult(
                case=case,
                status="error",
                failure=EvaluationFailure(
                    failure_type=_failure_type(exc),
                    error_code=exc.code.value if isinstance(exc, AppError) else None,
                    message=exc.message if isinstance(exc, AppError) else str(exc),
                ),
                trace_id=exc.trace_id if isinstance(exc, AppError) else None,
            )

        answer = execution.answer
        retrieved_chunks = tuple(
            EvaluationChunkObservation(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                version_id=chunk.version_id,
                rank=chunk.rank,
                score=chunk.score,
                retriever=chunk.retriever,
            )
            for chunk in execution.retrieval_result.results
        )
        return EvaluationCaseResult(
            case=case,
            status="success",
            retrieved_chunks=retrieved_chunks,
            context_chunk_ids=tuple(
                citation.chunk_id for citation in execution.packed_context.citations
            ),
            context_doc_ids=tuple(
                citation.doc_id for citation in execution.packed_context.citations
            ),
            context_source_paths=tuple(
                citation.source_path for citation in execution.packed_context.citations
            ),
            dropped_chunk_ids=tuple(
                dropped.chunk_id for dropped in execution.packed_context.dropped_chunks
            ),
            answer_status=answer.status,
            answer_text=answer.answer if self.config.include_answer_text else None,
            answer_citation_ids=tuple(citation.citation_id for citation in answer.citations),
            answer_citation_chunk_ids=tuple(citation.chunk_id for citation in answer.citations),
            answer_citation_doc_ids=tuple(citation.doc_id for citation in answer.citations),
            answer_citation_source_paths=tuple(
                citation.source_path for citation in answer.citations
            ),
            abstention_reason=answer.abstention_reason,
            trace_id=answer.trace_id,
            latency_ms=answer.latency_ms,
            metrics=calculate_case_metrics(
                case,
                retrieved_chunks=execution.retrieval_result.results,
                packed_context=execution.packed_context,
                answer=answer,
            ),
        )


@dataclass(slots=True)
class EvaluationSuiteRunner:
    """让同一数据集和索引在多个实验 profile 下顺序运行。"""

    runner: EvaluationRunner
    index: RagIndex
    target_builder: Callable[[ExperimentConfig], EvaluationRagService]

    def run_all(self) -> tuple[EvaluationRunResult, ...]:
        """先加载一次数据集，再让每个 profile 在相同输入上执行。"""

        cases = self.runner.load_cases()
        return tuple(
            self.runner.run(
                target=EvaluationTarget(
                    experiment=experiment,
                    service=self.target_builder(experiment),
                ),
                index=self.index,
                cases=cases,
            )
            for experiment in self.runner.config.experiments
        )


def _build_index_snapshot(index: RagIndex) -> EvaluationIndexSnapshot:
    manifest = index.manifest
    return EvaluationIndexSnapshot(
        index_id=manifest.index_id,
        artifact_definition_hash=manifest.artifact_definition_hash,
        document_set_hash=manifest.document_set_hash,
        document_count=manifest.document_count,
        chunk_count=manifest.chunk_count,
        vector_count=manifest.vector_count,
    )


def _build_run_id(experiment_name: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    normalized_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in experiment_name.strip()
    )
    return f"{timestamp}_{normalized_name}_{uuid4().hex[:8]}"


def _failure_type(exc: AppError | OSError | RuntimeError | TypeError | ValueError) -> str:
    if isinstance(exc, AppError):
        return exc.code.value.lower()
    if isinstance(exc, OSError):
        return "infrastructure"
    if isinstance(exc, TypeError):
        return "contract"
    if isinstance(exc, ValueError):
        return "validation"
    return "runtime"
