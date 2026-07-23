"""LLM Client 注册表。"""

from __future__ import annotations

from collections.abc import Callable

from app.llm.base import LlmClient
from app.llm.config import LlmClientConfig
from app.llm.mock import MockLlmClient
from app.llm.openai_compatible import OpenAiCompatibleLlmClient


LlmClientBuilder = Callable[[LlmClientConfig, str | None], LlmClient]


class LlmClientRegistry:
    """按 provider 名称管理 LLM Client 构建器。"""

    def __init__(self) -> None:
        self._builders: dict[str, LlmClientBuilder] = {}

    def register(self, provider: str, builder: LlmClientBuilder) -> None:
        """注册一个 provider 构建器。"""

        normalized_provider = _normalize_provider(provider)
        if normalized_provider in self._builders:
            raise ValueError(f"LLM provider 已注册：{normalized_provider}")
        self._builders[normalized_provider] = builder

    def create(self, config: LlmClientConfig, *, api_key: str | None) -> LlmClient:
        """根据 Config 创建对应 Client。"""

        try:
            builder = self._builders[config.provider]
        except KeyError as exc:
            available = ", ".join(sorted(self._builders)) or "无"
            raise ValueError(
                f"未注册的 LLM provider：{config.provider}；可用 provider：{available}"
            ) from exc
        return builder(config, api_key)

    @property
    def providers(self) -> tuple[str, ...]:
        """返回已注册 provider 名称。"""

        return tuple(sorted(self._builders))


def build_default_llm_client_registry() -> LlmClientRegistry:
    """构建项目内置 LLM Client 注册表。"""

    registry = LlmClientRegistry()
    registry.register("mock", lambda _config, _api_key: MockLlmClient())
    registry.register(
        "openai_compatible",
        lambda config, api_key: OpenAiCompatibleLlmClient(
            config=config,
            api_key=api_key or "",
        ),
    )
    return registry


def _normalize_provider(provider: str) -> str:
    """清理并校验 provider 名称。"""

    normalized_provider = provider.strip()
    if not normalized_provider:
        raise ValueError("LLM provider 不能为空")
    return normalized_provider

