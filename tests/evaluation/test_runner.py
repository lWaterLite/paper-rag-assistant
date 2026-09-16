"""评测运行器与本地运行产物测试。"""

from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from app.core.tracing import RagTrace
from app.evaluation.configuration import EvaluationConfig, ExperimentConfig
from app.evaluation.datasets import JsonlEvaluationDatasetRepository
from app.evaluation.reporting import EvaluationReportWriter
from app.evaluation.repositories import LocalJsonEvaluationResultRepository
from app.evaluation.runner import EvaluationRunner, EvaluationTarget
from app.generation.models import Citation, RagAnswer
from app.pipeline import RagPipelineExecutionResult
from app.retrieval.context.packer import (
    ContextCitation,
    ContextTokenUsage,
    PackedContext,
)
from app.retrieval.models import RetrievedChunk
from app.retrieval.pipeline import RetrievalPipelineResult
from app.retrieval.query import QueryPlan


class EvaluationRunnerTest(unittest.TestCase):
    """验证运行器复用生产观察边界并写出完整审计产物。"""

    def setUp(self) -> None:
        self._temp_dir = Path(".tmp_tests") / f"evaluation_runner_{uuid4().hex}"
        self._temp_dir.mkdir(parents=True)
        self._dataset_path = self._temp_dir / "dataset.jsonl"
        self._dataset_path.write_text(
            '{"id": "fact-001", "question": "主要结论是什么？", '
            '"question_type": "fact", "expected_doc_ids": ["paper-a"]}\n',
            encoding="utf-8",
        )
        self._config = EvaluationConfig(
            dataset_path=self._dataset_path,
            dataset_id="test_dataset",
            dataset_version="v1",
            output_dir=self._temp_dir / "runs",
            include_answer_text=True,
            max_cases=None,
            experiments=(
                ExperimentConfig(
                    name="vector_baseline",
                    retriever="vector",
                    top_k=3,
                    reranking_enabled=False,
                ),
            ),
        )
        self._runner = EvaluationRunner(
            config=self._config,
            dataset_repository=JsonlEvaluationDatasetRepository(),
            result_repository=LocalJsonEvaluationResultRepository(),
            report_writer=EvaluationReportWriter(),
        )

    def tearDown(self) -> None:
        shutil.rmtree(self._temp_dir, ignore_errors=True)

    def test_run_persists_case_facts_summary_and_markdown_report(self) -> None:
        result = self._run(FakeEvaluationService())

        self.assertEqual(result.run.summary.total_case_count, 1)
        self.assertEqual(result.run.summary.success_case_count, 1)
        self.assertEqual(result.run.summary.metrics["retrieval_hit_rate"].value, 1.0)
        self.assertTrue(Path(result.artifacts.run_path).is_file())
        self.assertTrue(Path(result.artifacts.case_results_path).is_file())
        self.assertTrue(Path(result.artifacts.summary_path).is_file())
        self.assertTrue(Path(result.artifacts.failures_path).is_file())
        report_path = Path(result.artifacts.report_path)
        self.assertTrue(report_path.is_file())
        self.assertIn("## 自动指标", report_path.read_text(encoding="utf-8"))

        summary = json.loads(Path(result.artifacts.summary_path).read_text(encoding="utf-8"))
        self.assertEqual(summary["success_case_count"], 1)

    def test_run_records_explicit_failure_without_aborting_other_cases(self) -> None:
        result = self._run(FailingEvaluationService())

        self.assertEqual(result.run.summary.error_case_count, 1)
        self.assertEqual(result.run.case_results[0].status, "error")
        self.assertEqual(result.run.summary.failure_counts, {"runtime": 1})
        failures = Path(result.artifacts.failures_path).read_text(encoding="utf-8")
        self.assertIn("服务暂时不可用", failures)

    def _run(self, service: FakeEvaluationService | FailingEvaluationService):
        cases = self._runner.load_cases()
        return self._runner.run(
            target=EvaluationTarget(
                experiment=self._config.experiments[0],
                service=service,
            ),
            index=_index(),
            cases=cases,
        )


class FakeEvaluationService:
    """返回固定完整执行结果的受控 RAG 服务替身。"""

    def execute(
        self,
        question: str,
        *,
        top_k: int | None = None,
        retriever: str | None = None,
    ) -> RagPipelineExecutionResult:
        _ = (question, top_k, retriever)
        chunk = _chunk()
        trace = RagTrace()
        trace.mark_success()
        context_citation = ContextCitation(
            citation_id="C1",
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            version_id=chunk.version_id,
            title=chunk.title,
            source_path=chunk.source_path,
            snippet=chunk.text,
        )
        packed_context = PackedContext(
            context_text=chunk.text,
            citations=[context_citation],
            used_chunks=[chunk],
            dropped_chunks=[],
            segments=[],
            token_usage=ContextTokenUsage(
                estimator="regex",
                question_tokens=1,
                reserved_prompt_tokens=1,
                reserved_output_tokens=1,
                safety_margin_tokens=1,
                available_context_tokens=100,
                used_context_tokens=1,
            ),
        )
        answer = RagAnswer(
            answer="测试回答",
            citations=[
                Citation.from_context_citation(context_citation),
            ],
            retrieved_chunks=[chunk],
            trace_id=trace.trace_id,
            latency_ms=trace.latency_ms,
            trace=trace,
        )
        return RagPipelineExecutionResult(
            answer=answer,
            query_plan=QueryPlan(original_query="测试问题", primary_query="测试问题"),
            retrieval_result=RetrievalPipelineResult(
                query="测试问题",
                retriever="vector",
                candidate_limit=3,
                top_k=3,
                results=[chunk],
                trace=trace,
            ),
            packed_context=packed_context,
        )


class FailingEvaluationService:
    """模拟可预期的运行时服务失败。"""

    def execute(
        self,
        question: str,
        *,
        top_k: int | None = None,
        retriever: str | None = None,
    ) -> RagPipelineExecutionResult:
        _ = (question, top_k, retriever)
        raise RuntimeError("服务暂时不可用")


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-a",
        doc_id="paper-a",
        content_hash="hash-a",
        version_id="v1",
        text="论文结论的证据。",
        score=0.9,
        rank=1,
        retriever="vector",
        source_path="data/raw/paper-a.md",
        chunk_index=0,
        title="Paper A",
    )


def _index() -> SimpleNamespace:
    return SimpleNamespace(
        manifest=SimpleNamespace(
            index_id="index-test",
            artifact_definition_hash="artifact-hash",
            document_set_hash="document-hash",
            document_count=1,
            chunk_count=1,
            vector_count=1,
        )
    )


if __name__ == "__main__":
    unittest.main()
