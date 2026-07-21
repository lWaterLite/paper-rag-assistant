"""候选证据变换阶段测试。"""

from __future__ import annotations

import unittest

from app.core.errors import AppError, ErrorCode
from app.retrieval.models import RetrievedChunk
from app.retrieval.context.evidence_transformers import (
    EvidenceCandidate,
    EvidenceSource,
    EvidenceTransformRequest,
    EvidenceTransformResult,
    EvidenceTransformationConfig,
)
from app.retrieval.context.evidence_transformers.registry import (
    build_default_evidence_transformer_registry,
)
from app.retrieval.context.evidence_transformers.stage import EvidenceTransformStage


def build_chunk(chunk_id: str, text: str, *, rank: int = 1) -> RetrievedChunk:
    """构造具有稳定来源信息的检索结果。"""

    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc_test",
        content_hash=f"hash_{chunk_id}",
        version_id="v1",
        text=text,
        score=1.0,
        rank=rank,
        retriever="test",
        source_path="test.md",
        chunk_index=rank - 1,
    )


class RaisingTransformer:
    """用于验证故障策略的 transformer。"""

    name = "raising"

    def transform(self, request: EvidenceTransformRequest) -> EvidenceTransformResult:
        _ = request
        raise RuntimeError("变换服务不可用")


class DroppingTransformer:
    """用于验证来源覆盖约束的 transformer。"""

    name = "dropping"

    def transform(self, request: EvidenceTransformRequest) -> EvidenceTransformResult:
        first_chunk = request.chunks[0]
        candidate = EvidenceCandidate(
            evidence_id="only_first",
            text=first_chunk.text,
            sources=(EvidenceSource.full_chunk(first_chunk),),
        )
        return EvidenceTransformResult(candidates=(candidate,))


class EvidenceTransformStageTest(unittest.TestCase):
    """验证当前 passthrough 与未来策略共享的阶段契约。"""

    def test_passthrough_strategy_preserves_full_text_and_source_range(self) -> None:
        chunk = build_chunk("chunk_1", "候选证据全文")
        registry = build_default_evidence_transformer_registry()
        stage = EvidenceTransformStage(
            config=EvidenceTransformationConfig(),
            transformer=registry.create(EvidenceTransformationConfig()),
        )

        result = stage.process(EvidenceTransformRequest(query="什么是 RAG", chunks=(chunk,)))

        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        self.assertEqual(candidate.text, chunk.text)
        self.assertEqual(candidate.sources[0].chunk, chunk)
        self.assertEqual(candidate.sources[0].char_start, 0)
        self.assertEqual(candidate.sources[0].char_end, len(chunk.text))
        self.assertFalse(result.detail["degraded"])

    def test_fail_open_falls_back_to_passthrough_candidates(self) -> None:
        chunk = build_chunk("chunk_1", "原始候选")
        stage = EvidenceTransformStage(
            config=EvidenceTransformationConfig(failure_mode="fail_open"),
            transformer=RaisingTransformer(),
        )

        result = stage.process(EvidenceTransformRequest(query="问题", chunks=(chunk,)))

        self.assertTrue(result.detail["degraded"])
        self.assertEqual(result.candidates[0].text, "原始候选")
        self.assertEqual(result.candidates[0].sources[0].char_end, len("原始候选"))

    def test_fail_closed_rejects_transformer_that_loses_source_coverage(self) -> None:
        first = build_chunk("chunk_1", "第一条证据", rank=1)
        second = build_chunk("chunk_2", "第二条证据", rank=2)
        stage = EvidenceTransformStage(
            config=EvidenceTransformationConfig(failure_mode="fail_closed"),
            transformer=DroppingTransformer(),
        )

        with self.assertRaises(AppError) as context:
            stage.process(
                EvidenceTransformRequest(query="问题", chunks=(first, second))
            )

        self.assertEqual(context.exception.code, ErrorCode.EVIDENCE_TRANSFORM_FAILED)
        self.assertIn("没有保留全部候选来源", context.exception.message)


if __name__ == "__main__":
    unittest.main()
