"""ConfigFactory 配置快照生命周期测试。"""

from __future__ import annotations

import unittest

from app.core.settings import EnvSettings, ProjectSettings
from app.factory.configs import ConfigFactory


class ConfigFactoryTest(unittest.TestCase):
    """验证 Config 只在单个应用组合根内转换和复用。"""

    @staticmethod
    def build_factory() -> ConfigFactory:
        """构造不依赖外部环境的独立配置工厂。"""

        return ConfigFactory(
            env_settings=EnvSettings(),
            project_settings=ProjectSettings(),
        )

    def test_reuses_same_config_instances_within_one_factory(self) -> None:
        factory = self.build_factory()

        self.assertIs(factory.build_loader_config(), factory.build_loader_config())
        self.assertIs(factory.build_chunker_config(), factory.build_chunker_config())
        self.assertIs(factory.build_embedding_config(), factory.build_embedding_config())
        self.assertIs(factory.build_llm_client_config(), factory.build_llm_client_config())
        self.assertIs(factory.build_generation_config(), factory.build_generation_config())
        self.assertIs(factory.build_retrieval_config(), factory.build_retrieval_config())
        self.assertIs(
            factory.build_postprocessing_config(),
            factory.build_postprocessing_config(),
        )

    def test_composed_configs_refer_to_the_same_domain_snapshot(self) -> None:
        factory = self.build_factory()
        postprocessing_config = factory.build_postprocessing_config()

        self.assertIs(postprocessing_config.retrieval, factory.build_retrieval_config())
        self.assertIs(
            postprocessing_config.reranking,
            factory.build_reranking_config(),
        )
        self.assertIs(
            postprocessing_config.context_packing,
            factory.build_context_packer_config(),
        )

    def test_does_not_share_config_instances_between_factories(self) -> None:
        first_factory = self.build_factory()
        second_factory = self.build_factory()

        self.assertIsNot(
            first_factory.build_embedding_config(),
            second_factory.build_embedding_config(),
        )
        self.assertIsNot(
            first_factory.build_postprocessing_config(),
            second_factory.build_postprocessing_config(),
        )
        self.assertIsNot(
            first_factory.build_generation_config(),
            second_factory.build_generation_config(),
        )


if __name__ == "__main__":
    unittest.main()
