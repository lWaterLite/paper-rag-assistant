"""LLM 基础设施对象组装。"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.errors import AppError, ErrorCode
from app.core.settings import EnvSettings
from app.factory.configs import ConfigFactory
from app.llm import (
    LlmClient,
    LlmClientRegistry,
    RetryingLlmClient,
    build_default_llm_client_registry,
)


@dataclass(slots=True)
class LlmFactory:
    """创建并缓存当前 ApplicationFactory 生命周期内的 LLM Client。"""

    configs: ConfigFactory
    env_settings: EnvSettings
    registry: LlmClientRegistry = field(default_factory=build_default_llm_client_registry)
    _client: LlmClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        config = self.configs.generation.llm
        secret = self.env_settings.llm_api_key
        api_key = secret.get_secret_value() if secret is not None else None
        try:
            client = self.registry.create(config, api_key=api_key)
        except ValueError as exc:
            raise AppError(ErrorCode.INVALID_CONFIG, str(exc)) from exc
        self._client = RetryingLlmClient(client, max_retries=config.max_retries)

    @property
    def client(self) -> LlmClient:
        """返回当前应用生命周期内复用的 LLM Client。"""

        return self._client
