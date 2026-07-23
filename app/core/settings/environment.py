"""环境变量与敏感配置。"""

from __future__ import annotations

from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.errors import AppError, ErrorCode
from app.core.settings.project import format_validation_error


class EnvSettings(BaseSettings):
    """从环境变量读取敏感配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        validate_default=True,
    )

    llm_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="RAG_LLM_API_KEY",
        description="外部 LLM provider 使用的 API key",
    )

    @classmethod
    def from_env(cls) -> "EnvSettings":
        """保留敏感配置加载入口，并转换为项目统一错误。"""

        try:
            return cls()
        except ValidationError as exc:
            raise AppError(
                ErrorCode.INVALID_CONFIG,
                format_validation_error(exc),
            ) from exc
