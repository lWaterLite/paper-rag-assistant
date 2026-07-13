"""Token-aware ContextPacker 测试。"""

from __future__ import annotations

import unittest

from app.core.models import RetrievedChunk
from app.retrieval.context_packer import (
    ContextPackerConfig,
    ContextPackRequest,
    TokenAwareContextPacker,
)
from app.retrieval.token_estimators import RegexTokenEstimator


def build_retrieved_chunk(
    chunk_id: str,
    text: str,
    *,
    doc_id: str = "doc_test",
    version_id: str = "v_test",
    chunk_index: int = 0,
    rank: int = 1,
) -> RetrievedChunk:
    """构造测试检索结果。"""

    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content_hash=f"hash_{doc_id}",
        version_id=version_id,
        text=text,
        score=1.0,
        rank=rank,
        retriever="test",
        source_path=f"{doc_id}.md",
        chunk_index=chunk_index,
        title="测试文档",
        section="测试小节",
        metadata={"source": "test"},
    )


def build_packer(**overrides: int) -> TokenAwareContextPacker:
    """构造带稳定 token 预算的测试 packer。"""

    config_values = {
        "model_context_window": 120,
        "max_context_tokens": 60,
        "reserved_prompt_tokens": 10,
        "reserved_output_tokens": 20,
        "safety_margin_tokens": 5,
        "max_chunks_per_document": 2,
    }
    config_values.update(overrides)
    config = ContextPackerConfig(**config_values)
    return TokenAwareContextPacker(
        config=config,
        token_estimator=RegexTokenEstimator(),
    )


class TokenAwareContextPackerTest(unittest.TestCase):
    """验证 token 预算、段级 provenance 与上下文选择策略。"""

    def test_pack_deduplicates_same_text(self) -> None:
        packed = build_packer().pack(
            ContextPackRequest(
                query="RAG",
                chunks=(
                    build_retrieved_chunk("chunk_1", "重复内容", chunk_index=0),
                    build_retrieved_chunk("chunk_2", "重复内容", chunk_index=1),
                ),
            )
        )

        self.assertEqual([chunk.chunk_id for chunk in packed.used_chunks], ["chunk_1"])
        self.assertEqual(packed.dropped_chunks[0].reason, "duplicate_content")

    def test_pack_merges_adjacent_chunks_and_preserves_all_source_chunk_ids(self) -> None:
        packed = build_packer().pack(
            ContextPackRequest(
                query="RAG",
                chunks=(
                    build_retrieved_chunk("chunk_1", "第一段", chunk_index=0),
                    build_retrieved_chunk("chunk_2", "第二段", chunk_index=1),
                ),
            )
        )

        self.assertEqual(len(packed.segments), 1)
        self.assertEqual(packed.segments[0].source_chunk_ids, ("chunk_1", "chunk_2"))
        self.assertIn("第一段\n第二段", packed.context_text)
        self.assertEqual(packed.citations[0].citation_id, "C1")

    def test_pack_applies_per_document_chunk_quota(self) -> None:
        packed = build_packer(max_chunks_per_document=1).pack(
            ContextPackRequest(
                query="RAG",
                chunks=(
                    build_retrieved_chunk("chunk_1", "第一段", chunk_index=0),
                    build_retrieved_chunk("chunk_2", "第二段", chunk_index=1),
                    build_retrieved_chunk(
                        "chunk_3",
                        "另一篇论文内容",
                        doc_id="doc_other",
                    ),
                ),
            )
        )

        self.assertEqual(
            [chunk.chunk_id for chunk in packed.used_chunks], ["chunk_1", "chunk_3"]
        )
        self.assertEqual(
            packed.dropped_chunks[0].reason,
            "document_chunk_quota_exceeded",
        )

    def test_pack_uses_token_budget_and_records_remaining_candidates(self) -> None:
        packer = build_packer(max_context_tokens=12)
        packed = packer.pack(
            ContextPackRequest(
                query="RAG",
                chunks=(
                    build_retrieved_chunk("chunk_1", "alpha beta", chunk_index=0),
                    build_retrieved_chunk(
                        "chunk_2",
                        "gamma delta epsilon zeta eta theta",
                        doc_id="doc_other",
                    ),
                ),
            )
        )

        self.assertLessEqual(
            packed.token_usage.used_context_tokens,
            packed.token_usage.available_context_tokens,
        )
        self.assertTrue(
            packed.dropped_chunks
            or any(segment.is_truncated for segment in packed.segments)
        )

    def test_pack_truncates_long_candidate_with_source_provenance(self) -> None:
        packed = build_packer(max_context_tokens=12).pack(
            ContextPackRequest(
                query="RAG",
                chunks=(
                    build_retrieved_chunk(
                        "chunk_1",
                        "alpha beta gamma delta epsilon zeta eta theta iota kappa",
                    ),
                ),
            )
        )

        self.assertEqual(len(packed.segments), 1)
        self.assertTrue(packed.segments[0].is_truncated)
        self.assertEqual(packed.segments[0].source_chunk_ids, ("chunk_1",))
        self.assertTrue(packed.segments[0].text.endswith("..."))

    def test_question_and_reserved_tokens_reduce_available_context_budget(self) -> None:
        packed = build_packer(
            model_context_window=30,
            max_context_tokens=25,
            reserved_prompt_tokens=8,
            reserved_output_tokens=8,
            safety_margin_tokens=2,
        ).pack(
            ContextPackRequest(
                query="问题内容",
                chunks=(build_retrieved_chunk("chunk_1", "alpha beta gamma"),),
            )
        )

        self.assertLess(packed.token_usage.available_context_tokens, 25)
        self.assertEqual(packed.token_usage.estimator, "regex")


if __name__ == "__main__":
    unittest.main()
