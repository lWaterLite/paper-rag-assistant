"""检索后处理流程的组合配置、校验与摘要测试。"""

from __future__ import annotations

import unittest

from app.core.errors import AppError, ErrorCode
from app.core.settings import (
    ContextPackingSettings,
    EnvSettings,
    ProjectSettings,
    RerankingSettings,
    RetrievalSettings,
)
from app.factory import ApplicationFactory
from app.retrieval.configuration import RetrievalConfig
from app.retrieval.configuration.postprocessing import (
    PostProcessingConfig,
    PostProcessingConfigValidator,
    PostProcessingProfile,
)
from app.retrieval.context import ContextPackerConfig
from app.retrieval.context.evidence_transformers import EvidenceTransformationConfig
from app.retrieval.rerankers import RerankingConfig


class PostProcessingConfigTest(unittest.TestCase):
    """验证后处理配置只处理跨模块的组合规则。"""

    def test_profile_describes_enabled_rerank_and_effective_context_budget(self) -> None:
        config = PostProcessingConfig(
            retrieval=RetrievalConfig(top_k=3),
            reranking=RerankingConfig(
                enabled=True,
                candidate_limit=12,
                failure_mode="fail_open",
            ),
            context_packing=ContextPackerConfig(
                model_context_window=4096,
                max_context_tokens=1800,
                reserved_prompt_tokens=200,
                reserved_output_tokens=512,
                safety_margin_tokens=64,
                max_chunks_per_document=2,
            ),
            evidence_transformation=EvidenceTransformationConfig(),
        )

        PostProcessingConfigValidator.validate(config)
        profile = PostProcessingProfile.from_config(config)

        self.assertTrue(profile.reranking_enabled)
        self.assertEqual(profile.reranking_strategy, "lexical")
        self.assertEqual(profile.candidate_limit_source, "reranking_candidate_window")
        self.assertEqual(profile.default_candidate_limit, 12)
        self.assertEqual(profile.static_available_context_tokens, 3320)
        self.assertEqual(profile.effective_context_token_limit, 1800)

    def test_disabled_rerank_uses_resolved_top_k_as_candidate_limit(self) -> None:
        config = PostProcessingConfig(
            retrieval=RetrievalConfig(top_k=4),
            reranking=RerankingConfig(enabled=False, candidate_limit=12),
            context_packing=ContextPackerConfig(),
            evidence_transformation=EvidenceTransformationConfig(),
        )

        PostProcessingConfigValidator.validate(config)
        profile = PostProcessingProfile.from_config(config)

        self.assertFalse(profile.reranking_enabled)
        self.assertIsNone(profile.reranking_strategy)
        self.assertIsNone(profile.configured_candidate_limit)
        self.assertEqual(profile.candidate_limit_source, "resolved_top_k")
        self.assertEqual(profile.default_candidate_limit, 4)

    def test_validator_rejects_rerank_window_smaller_than_default_top_k(self) -> None:
        config = PostProcessingConfig(
            retrieval=RetrievalConfig(top_k=5),
            reranking=RerankingConfig(enabled=True, candidate_limit=3),
            context_packing=ContextPackerConfig(),
            evidence_transformation=EvidenceTransformationConfig(),
        )

        with self.assertRaisesRegex(ValueError, "reranking.candidate_limit"):
            PostProcessingConfigValidator.validate(config)

    def test_validator_rejects_context_budget_larger_than_static_capacity(self) -> None:
        config = PostProcessingConfig(
            retrieval=RetrievalConfig(top_k=3),
            reranking=RerankingConfig(enabled=False),
            context_packing=ContextPackerConfig(
                model_context_window=1000,
                max_context_tokens=600,
                reserved_prompt_tokens=200,
                reserved_output_tokens=200,
                safety_margin_tokens=100,
                max_chunks_per_document=2,
            ),
            evidence_transformation=EvidenceTransformationConfig(),
        )

        with self.assertRaisesRegex(ValueError, "max_context_tokens"):
            PostProcessingConfigValidator.validate(config)

    def test_factory_rejects_invalid_combination_before_constructing_context_packer(self) -> None:
        factory = ApplicationFactory(
            env_settings=EnvSettings(),
            project_settings=ProjectSettings(
                retrieval=RetrievalSettings(
                    top_k=5,
                    reranking=RerankingSettings(enabled=True, candidate_limit=3),
                    context_packing=ContextPackingSettings(),
                )
            ),
        )

        with self.assertRaises(AppError) as context:
            factory.retrieval.build_context_packer()

        self.assertEqual(context.exception.code, ErrorCode.INVALID_CONFIG)
        self.assertIn("reranking.candidate_limit", context.exception.message)


if __name__ == "__main__":
    unittest.main()
