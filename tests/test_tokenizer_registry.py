"""分词器策略与注册表测试。"""

from __future__ import annotations

import unittest

from app.core.settings import (
    EnvSettings,
    ProjectSettings,
    RetrievalSettings,
    TokenizerSettings,
)
from app.factory.configs import ConfigFactory
from app.factory.retrieval import RetrievalFactory
from app.retrieval.tokenizers import (
    RegexTokenizer,
    TokenizerConfig,
    TokenizerRegistry,
    build_default_tokenizer_registry,
)


class TokenizerRegistryTest(unittest.TestCase):
    """验证分词器策略注册与创建。"""

    def test_default_registry_creates_regex_tokenizer(self) -> None:
        registry = build_default_tokenizer_registry()

        tokenizer = registry.create(TokenizerConfig(strategy="regex"))

        self.assertIsInstance(tokenizer, RegexTokenizer)
        self.assertEqual(tokenizer.tokenize("RAG 检索"), ["rag", "检", "索"])

    def test_registry_supports_external_tokenizer_provider(self) -> None:
        class ExternalTokenizer:
            @staticmethod
            def tokenize(text: str) -> list[str]:
                return [text]

        registry = TokenizerRegistry()
        registry.register("external", ExternalTokenizer)

        tokenizer = registry.create(TokenizerConfig(strategy="external"))

        self.assertIsInstance(tokenizer, ExternalTokenizer)

    def test_registry_rejects_unknown_strategy(self) -> None:
        registry = build_default_tokenizer_registry()

        with self.assertRaisesRegex(ValueError, "未知 tokenizer strategy"):
            registry.create(TokenizerConfig(strategy="missing"))

    def test_retrieval_factory_selects_tokenizer_from_project_settings(self) -> None:
        class ExternalTokenizer:
            @staticmethod
            def tokenize(text: str) -> list[str]:
                return [text]

        registry = TokenizerRegistry()
        registry.register("external", ExternalTokenizer)
        project_settings = ProjectSettings(
            retrieval=RetrievalSettings(
                tokenizer=TokenizerSettings(strategy="external")
            )
        )
        factory = RetrievalFactory(
            configs=ConfigFactory(
                env_settings=EnvSettings(),
                project_settings=project_settings,
            ),
            tokenizer_registry=registry,
        )

        tokenizer = factory.build_tokenizer()

        self.assertIsInstance(tokenizer, ExternalTokenizer)


if __name__ == "__main__":
    unittest.main()
