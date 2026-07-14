"""Embedding Provider 注册表。"""

from __future__ import annotations

from collections.abc import Callable

from app.indexing.configuration import EmbeddingConfig
from app.indexing.embeddings.base import EmbeddingClient
from app.indexing.embeddings.mock import MockEmbeddingClient
from app.indexing.embeddings.openai import OpenAIEmbeddingClient

EmbeddingClientBuilder = Callable[[EmbeddingConfig, str | None], EmbeddingClient]


class EmbeddingClientRegistry:
    """按 provider 名称创建 embedding 客户端。"""

    def __init__(self) -> None:
        self._builders: dict[str, EmbeddingClientBuilder] = {}

    def register(
        self,
        provider: str,
        builder: EmbeddingClientBuilder,
        *,
        replace: bool = False,
    ) -> None:
        """注册 provider 的客户端构造器。"""

        normalized_provider = _normalize_provider(provider)
        if normalized_provider in self._builders and not replace:
            raise ValueError(f"embedding provider 已注册：{normalized_provider}")
        self._builders[normalized_provider] = builder

    def create(
        self,
        config: EmbeddingConfig,
        *,
        api_key: str | None = None,
    ) -> EmbeddingClient:
        """根据运行时配置创建 embedding 客户端。"""

        provider = _normalize_provider(config.provider)
        builder = self._builders.get(provider)
        if builder is None:
            available = ", ".join(sorted(self._builders)) or "无"
            raise ValueError(
                f"不支持的 embedding provider：{provider}；已注册：{available}"
            )
        return builder(config, api_key)


def build_default_embedding_client_registry() -> EmbeddingClientRegistry:
    """创建项目内置 embedding provider 注册表。"""

    registry = EmbeddingClientRegistry()
    registry.register("mock", lambda config, _: MockEmbeddingClient(config))
    registry.register(
        "openai",
        lambda config, api_key: OpenAIEmbeddingClient(config, api_key=api_key),
    )
    return registry


def _normalize_provider(provider: str) -> str:
    """规范 provider 名称并拒绝空白值。"""

    normalized = provider.strip().lower()
    if not normalized:
        raise ValueError("embedding provider 不能为空")
    return normalized
