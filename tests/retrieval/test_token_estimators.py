"""Token estimator 策略与注册表测试。"""

from __future__ import annotations

import unittest

from app.retrieval.context.token_estimators import TokenEstimatorConfig, TokenEstimatorRegistry
from app.retrieval.context.token_estimators.regex import RegexTokenEstimator
from app.retrieval.context.token_estimators.registry import (
    build_default_token_estimator_registry,
)


class TokenEstimatorTest(unittest.TestCase):
    """验证模型窗口预算与 BM25 tokenizer 采用独立策略对象。"""

    def test_regex_estimator_counts_words_cjk_and_punctuation(self) -> None:
        estimator = RegexTokenEstimator()

        self.assertEqual(estimator.count_text("RAG 检索, test!"), 6)

    def test_registry_creates_configured_estimator(self) -> None:
        estimator = build_default_token_estimator_registry().create(
            TokenEstimatorConfig(strategy="regex")
        )

        self.assertEqual(estimator.name, "regex")

    def test_registry_rejects_unknown_strategy(self) -> None:
        registry = TokenEstimatorRegistry()

        with self.assertRaises(ValueError):
            registry.create(TokenEstimatorConfig(strategy="unknown"))


if __name__ == "__main__":
    unittest.main()
