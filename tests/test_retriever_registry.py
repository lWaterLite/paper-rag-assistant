"""RetrieverRegistry 测试。"""

from __future__ import annotations

import unittest
from typing import cast

from app.core.models import RetrievedChunk
from app.core.settings import EnvSettings, ProjectSettings, RetrievalSettings
from app.factory.configs import ConfigFactory
from app.factory.retrieval import RetrievalFactory
from app.indexing.index_builder import RagIndex
from app.retrieval.retrievers import RetrieverRegistry


class EmptyRetriever:
    """测试用空检索器。"""

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        _ = query, top_k
        return []


class RetrieverRegistryTest(unittest.TestCase):
    """验证检索策略注册、惰性创建和外部扩展。"""

    def test_resolve_creates_provider_result_only_once(self) -> None:
        registry = RetrieverRegistry()
        build_count = 0

        def provider() -> EmptyRetriever:
            nonlocal build_count
            build_count += 1
            return EmptyRetriever()

        registry.register("external", provider)

        first = registry.resolve("external")
        second = registry.resolve("external")

        self.assertIs(first, second)
        self.assertEqual(build_count, 1)

    def test_registry_detects_provider_cycle(self) -> None:
        registry = RetrieverRegistry()
        registry.register("cycle", lambda: registry.resolve("cycle"))

        with self.assertRaisesRegex(RuntimeError, "循环依赖"):
            registry.resolve("cycle")

    def test_factory_resolves_external_strategy_without_if_branch(self) -> None:
        project_settings = ProjectSettings(
            retrieval=RetrievalSettings(strategy="external")
        )
        factory = RetrievalFactory(
            configs=ConfigFactory(
                env_settings=EnvSettings(),
                project_settings=project_settings,
            )
        )
        registry = RetrieverRegistry()
        registry.register("external", EmptyRetriever)

        retriever = factory.build_retriever(
            cast(RagIndex, object()),
            registry=registry,
        )

        self.assertIsInstance(retriever, EmptyRetriever)

    def test_registry_reports_supported_strategies(self) -> None:
        registry = RetrieverRegistry()
        registry.register("vector", EmptyRetriever)

        with self.assertRaisesRegex(ValueError, "vector"):
            registry.resolve("missing")


if __name__ == "__main__":
    unittest.main()
