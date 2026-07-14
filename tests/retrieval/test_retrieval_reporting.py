"""Retrieval 子系统报告测试。"""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from app.core.errors import AppError, ErrorCode
from app.core.models import RetrievedChunk
from app.core.settings import (
    EnvSettings,
    ProjectSettings,
    RetrievalReportSettings,
    RetrievalSettings,
)
from app.factory import ApplicationFactory
from app.retrieval.configuration import RetrievalConfig
from app.retrieval.configuration.postprocessing import (
    PostProcessingConfig,
    PostProcessingProfile,
)
from app.retrieval.context import ContextPackerConfig
from app.retrieval.context.evidence_transformers import EvidenceTransformationConfig
from app.retrieval.reporting import (
    RetrievalConfigSnapshot,
    RetrievalIndexSnapshot,
    RetrievalReportConfig,
    RetrievalReporter,
    RetrievalReportWriter,
    RetrievalRuntimeSnapshot,
)
from app.retrieval.retrievers import RetrieverRegistry
from app.retrieval.rerankers import LexicalReranker, RerankingConfig
from app.retrieval.tokenizers import RegexTokenizer
from app.retrieval.services.search import SearchService


class StaticRetriever:
    """返回预设结果的测试检索器。"""

    def __init__(self, results: list[RetrievedChunk]) -> None:
        self._results = results

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        _ = query, top_k
        return list(self._results)


class FailingRetriever:
    """用于验证失败报告的测试检索器。"""

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        _ = query, top_k
        raise RuntimeError("测试检索失败")


def build_result(chunk_id: str, rank: int) -> RetrievedChunk:
    """创建测试检索结果。"""

    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        content_hash=f"hash_{chunk_id}",
        version_id=f"version_{chunk_id}",
        text=f"{chunk_id} 的完整文本",
        score=1.0 / rank,
        rank=rank,
        retriever="vector",
        source_path=f"docs/{chunk_id}.md",
        chunk_index=rank - 1,
    )


def build_runtime_snapshot() -> RetrievalRuntimeSnapshot:
    """创建稳定的测试运行时快照。"""

    return RetrievalRuntimeSnapshot(
        index=RetrievalIndexSnapshot(
            index_id="idx_test",
            schema_version=3,
            status="ready",
            config_hash="config_hash",
            document_set_hash="document_hash",
            document_count=2,
            chunk_count=3,
            vector_count=3,
            embedding_provider="mock",
            embedding_model="mock-hash-embedding",
            embedding_dimension=16,
            vector_repository_type="memory",
            vector_collection_name="test",
            distance_metric="cosine",
        ),
        config=RetrievalConfigSnapshot(
            default_strategy="vector",
            default_top_k=2,
            deduplicate_by_chunk_id=True,
            tokenizer_strategy="regex",
            bm25_k1=1.5,
            bm25_b=0.75,
            hybrid_candidate_multiplier=3,
            hybrid_rrf_rank_constant=60,
            hybrid_vector_weight=1.0,
            hybrid_bm25_weight=1.0,
            postprocessing=PostProcessingProfile.from_config(
                PostProcessingConfig(
                    retrieval=RetrievalConfig(strategy="vector", top_k=2),
                    reranking=RerankingConfig(enabled=False),
                    context_packing=ContextPackerConfig(),
                    evidence_transformation=EvidenceTransformationConfig(),
                )
            ),
            registered_strategies=("bm25", "hybrid", "vector"),
        ),
    )


def build_reporter(output_dir: Path) -> RetrievalReporter:
    """创建启用写入的测试 reporter。"""

    reporter = RetrievalReporter(
        config=RetrievalReportConfig(enabled=True, output_dir=output_dir),
        runtime_snapshot=build_runtime_snapshot(),
        writer=RetrievalReportWriter(),
    )
    reporter.prepare_output_directory()
    return reporter


class RetrievalReportingTest(unittest.TestCase):
    """验证 retrieval 成功、失败和问答入口都产生报告。"""

    def test_success_report_contains_counts_stages_and_runtime_snapshot(self) -> None:
        output_dir = Path(".tmp_tests") / f"retrieval_{uuid.uuid4().hex}"
        registry = RetrieverRegistry()
        registry.register(
            "vector",
            lambda: StaticRetriever(
                [
                    build_result("a", 1),
                    build_result("a", 2),
                    build_result("b", 3),
                ]
            ),
        )
        service = SearchService(
            registry=registry,
            config=RetrievalConfig(strategy="vector", top_k=2),
            reranking_config=RerankingConfig(enabled=False),
            reranker=None,
            reporter=build_reporter(output_dir),
        )

        try:
            result = service.search("RAG report")
            self.assertIsNotNone(result.report_path)
            report_path = result.report_path
            if report_path is None:
                self.fail("启用 retrieval report 后没有返回报告路径")
            report = json.loads(report_path.read_text(encoding="utf-8"))

            self.assertEqual(report["status"], "success")
            self.assertEqual(report["counts"]["candidate_count"], 3)
            self.assertEqual(report["counts"]["deduplicated_count"], 2)
            self.assertEqual(report["counts"]["returned_count"], 2)
            self.assertEqual(report["runtime"]["index"]["index_id"], "idx_test")
            self.assertEqual(
                [stage["stage"] for stage in report["stages"]],
                [
                    "retriever_execution",
                    "ChunkIdDeduplicationStage",
                    "TopKLimitStage",
                ],
            )
            self.assertEqual(
                report["runtime"]["config"]["postprocessing"][
                    "candidate_limit_source"
                ],
                "resolved_top_k",
            )
            self.assertNotIn("text_preview", report["results"][0])
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_failed_retrieval_writes_final_failure_report(self) -> None:
        output_dir = Path(".tmp_tests") / f"retrieval_{uuid.uuid4().hex}"
        registry = RetrieverRegistry()
        registry.register("vector", FailingRetriever)
        service = SearchService(
            registry=registry,
            config=RetrievalConfig(strategy="vector", top_k=2),
            reranking_config=RerankingConfig(enabled=False),
            reranker=None,
            reporter=build_reporter(output_dir),
        )

        try:
            with self.assertRaises(AppError) as context:
                service.search("失败请求")

            report_path = output_dir / f"retrieval_{context.exception.trace_id}.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "error")
            self.assertEqual(report["trace"]["final_status"], "error")
            self.assertEqual(
                report["failure"]["error_code"],
                ErrorCode.RETRIEVAL_FAILED.value,
            )
            self.assertIn("测试检索失败", report["failure"]["error_message"])
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_rag_pipeline_uses_same_retrieval_reporting_component(self) -> None:
        output_dir = Path(".tmp_tests") / f"retrieval_{uuid.uuid4().hex}"
        project_settings = ProjectSettings(
            retrieval=RetrievalSettings(
                top_k=2,
                report=RetrievalReportSettings(
                    enabled=True,
                    output_dir=output_dir,
                ),
            )
        )
        factory = ApplicationFactory(
            env_settings=EnvSettings(),
            project_settings=project_settings,
        )

        try:
            index, _ = factory.build_index_builder().build_from_directory(
                Path("data/raw/papers")
            )
            factory.build_rag_pipeline(index).ask("RAG 为什么需要引用？")

            report_paths = list(output_dir.glob("retrieval_*.json"))
            self.assertEqual(len(report_paths), 1)
            report = json.loads(report_paths[0].read_text(encoding="utf-8"))
            self.assertEqual(report["report_type"], "retrieval_execution")
            self.assertEqual(report["status"], "success")
            self.assertEqual(report["runtime"]["index"]["index_id"], index.manifest.index_id)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_rerank_stage_report_records_candidate_limit_and_strategy(self) -> None:
        output_dir = Path(".tmp_tests") / f"retrieval_{uuid.uuid4().hex}"
        registry = RetrieverRegistry()
        registry.register(
            "vector",
            lambda: StaticRetriever(
                [
                    build_result("a", 1),
                    build_result("b", 2),
                    build_result("c", 3),
                ]
            ),
        )
        service = SearchService(
            registry=registry,
            config=RetrievalConfig(strategy="vector", top_k=1),
            reranking_config=RerankingConfig(
                enabled=True,
                candidate_limit=3,
                batch_size=2,
            ),
            reranker=LexicalReranker(RegexTokenizer(), batch_size=2),
            reporter=build_reporter(output_dir),
        )

        try:
            result = service.search("完整文本")
            if result.report_path is None:
                self.fail("启用 retrieval report 后没有返回报告路径")
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            rerank_stage = next(
                stage for stage in report["stages"] if stage["stage"] == "RerankStage"
            )

            self.assertEqual(report["request"]["resolved_candidate_limit"], 3)
            self.assertEqual(rerank_stage["detail"]["reranker"], "lexical")
            self.assertFalse(rerank_stage["detail"]["degraded"])
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
