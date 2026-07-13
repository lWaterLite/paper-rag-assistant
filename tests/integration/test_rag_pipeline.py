"""RAG 索引构建与在线问答流程测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

from app.core.settings import EnvSettings, ProjectSettings, RetrievalSettings
from app.core.errors import AppError, ErrorCode
from app.core.models import RagAnswer
from app.factory import ApplicationFactory
from app.retrieval.context import PackedContext


SAMPLE_SOURCE_DIR = Path("data/raw/papers")


class RagPipelineTest(unittest.TestCase):
    """验证离线索引流程与在线问答流程。"""

    def test_index_builder_builds_in_memory_index(self) -> None:
        env_settings = EnvSettings()
        project_settings = ProjectSettings()
        index, result = create_factory(env_settings, project_settings).build_index_builder().build_from_directory(SAMPLE_SOURCE_DIR)

        self.assertGreater(result.document_count, 0)
        self.assertGreater(result.chunk_count, 0)
        self.assertEqual(result.chunk_count, result.vector_count)
        self.assertEqual(result.vector_count, index.vector_collection.count())
        self.assertEqual(result.manifest.document_count, result.document_count)
        self.assertEqual(result.manifest.chunk_count, result.chunk_count)
        self.assertEqual(result.manifest.vector_count, result.vector_count)
        self.assertEqual(
            result.manifest.chunk_size,
            project_settings.ingestion.chunking.chunk_size,
        )
        self.assertEqual(
            result.manifest.chunk_overlap,
            project_settings.ingestion.chunking.chunk_overlap,
        )
        self.assertEqual(result.manifest.embedding_provider, "mock")
        self.assertGreater(len(result.manifest.document_versions), 0)

    def test_pipeline_returns_structured_answer(self) -> None:
        env_settings = EnvSettings()
        project_settings = ProjectSettings(
            retrieval=RetrievalSettings(top_k=2)
        )
        factory = create_factory(env_settings, project_settings)
        index, _ = factory.build_index_builder().build_from_directory(SAMPLE_SOURCE_DIR)
        answer = factory.build_rag_pipeline(index).ask("RAG 为什么需要引用？")

        self.assertTrue(answer.answer)
        self.assertTrue(answer.trace_id.startswith("trace_"))
        self.assertLessEqual(
            len(answer.retrieved_chunks),
            project_settings.retrieval.top_k,
        )
        self.assertGreater(len(answer.citations), 0)

    def test_pipeline_uses_configured_bm25_retriever(self) -> None:
        env_settings = EnvSettings()
        project_settings = ProjectSettings(
            retrieval=RetrievalSettings(strategy="bm25", top_k=2)
        )
        factory = create_factory(env_settings, project_settings)
        index, _ = factory.build_index_builder().build_from_directory(SAMPLE_SOURCE_DIR)

        answer = factory.build_rag_pipeline(index).ask("retrieval generation")

        self.assertGreater(len(answer.retrieved_chunks), 0)
        self.assertEqual(answer.retrieved_chunks[0].retriever, "bm25")

    def test_pipeline_uses_configured_hybrid_retriever(self) -> None:
        env_settings = EnvSettings()
        project_settings = ProjectSettings(
            retrieval=RetrievalSettings(strategy="hybrid", top_k=2)
        )
        factory = create_factory(env_settings, project_settings)
        index, _ = factory.build_index_builder().build_from_directory(
            SAMPLE_SOURCE_DIR
        )

        answer = factory.build_rag_pipeline(index).ask("retrieval generation")

        self.assertGreater(len(answer.retrieved_chunks), 0)
        self.assertEqual(answer.retrieved_chunks[0].retriever, "hybrid")
        self.assertGreater(
            len(answer.retrieved_chunks[0].retrieval_signals),
            0,
        )

    def test_pipeline_marks_trace_success_when_all_stages_succeed(self) -> None:
        env_settings = EnvSettings()
        factory = create_factory(env_settings)
        index, _ = factory.build_index_builder().build_from_directory(SAMPLE_SOURCE_DIR)
        answer_generator = CapturingAnswerGenerator()
        pipeline = factory.build_rag_pipeline(index, answer_generator=answer_generator)

        pipeline.ask("正常问题")

        self.assertIsNotNone(answer_generator.trace)
        self.assertEqual(answer_generator.trace.final_status, "success")
        self.assertIsNone(answer_generator.trace.failure_type)

    def test_pipeline_records_failure_trace_when_retrieval_fails(self) -> None:
        env_settings = EnvSettings()
        factory = create_factory(env_settings)
        index, _ = factory.build_index_builder().build_from_directory(SAMPLE_SOURCE_DIR)
        pipeline = factory.build_rag_pipeline(index, retriever=FailingRetriever())

        with self.assertRaises(AppError) as context:
            pipeline.ask("会失败的问题")

        error = context.exception
        self.assertEqual(error.code, ErrorCode.RETRIEVAL_FAILED)
        self.assertIsNotNone(error.trace_id)
        self.assertEqual(error.trace.final_status, "error")
        self.assertEqual(error.trace.failure_type, "retrieval")
        self.assertEqual(error.trace.stages[-1].stage, "retrieval")
        self.assertEqual(error.trace.stages[-1].status, "error")

    def test_pipeline_records_failure_trace_when_context_packing_fails(self) -> None:
        env_settings = EnvSettings()
        factory = create_factory(env_settings)
        index, _ = factory.build_index_builder().build_from_directory(SAMPLE_SOURCE_DIR)
        pipeline = factory.build_rag_pipeline(index, context_packer=FailingContextPacker())

        with self.assertRaises(AppError) as context:
            pipeline.ask("会失败的问题")

        error = context.exception
        self.assertEqual(error.code, ErrorCode.RETRIEVAL_FAILED)
        self.assertEqual(error.trace.final_status, "error")
        self.assertEqual(error.trace.failure_type, "context_packing")
        self.assertEqual([stage.stage for stage in error.trace.stages], ["retrieval", "context_packing"])

    def test_pipeline_records_failure_trace_when_generation_fails(self) -> None:
        env_settings = EnvSettings()
        factory = create_factory(env_settings)
        index, _ = factory.build_index_builder().build_from_directory(SAMPLE_SOURCE_DIR)
        pipeline = factory.build_rag_pipeline(index, answer_generator=FailingAnswerGenerator())

        with self.assertRaises(AppError) as context:
            pipeline.ask("会失败的问题")

        error = context.exception
        self.assertEqual(error.code, ErrorCode.GENERATION_FAILED)
        self.assertEqual(error.trace.final_status, "error")
        self.assertEqual(error.trace.failure_type, "generation")
        self.assertEqual([stage.stage for stage in error.trace.stages], ["retrieval", "context_packing", "generation"])


class FailingRetriever:
    """测试用失败检索器。"""

    def retrieve(self, query: str, top_k: int):
        raise AppError(ErrorCode.RETRIEVAL_FAILED, "检索失败")


class CapturingAnswerGenerator:
    """测试用回答生成器，用于检查成功 trace。"""

    def __init__(self) -> None:
        self.trace = None

    def generate(
        self,
        question: str,
        packed_context: PackedContext,
        retrieved_chunks: list,
        trace,
    ) -> RagAnswer:
        self.trace = trace
        return RagAnswer(
            answer="测试回答",
            citations=packed_context.citations,
            retrieved_chunks=retrieved_chunks,
            trace_id=trace.trace_id,
            latency_ms=trace.latency_ms,
        )


class FailingContextPacker:
    """测试用失败上下文组织器。"""

    def pack(self, request):
        _ = request
        raise RuntimeError("上下文组织失败")


class FailingAnswerGenerator:
    """测试用失败回答生成器。"""

    def generate(
        self,
        question: str,
        packed_context: PackedContext,
        retrieved_chunks: list,
        trace,
    ) -> RagAnswer:
        raise AppError(ErrorCode.GENERATION_FAILED, "生成失败")


def create_factory(
        env_settings: EnvSettings,
        project_settings: ProjectSettings | None = None,
) -> ApplicationFactory:
    """通过应用组合根创建测试依赖。"""

    return ApplicationFactory(
        env_settings=env_settings,
        project_settings=project_settings if project_settings is not None else ProjectSettings(),
    )


if __name__ == "__main__":
    unittest.main()
