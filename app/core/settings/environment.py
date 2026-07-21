"""环境变量与敏感配置。"""

from __future__ import annotations

from pydantic import Field, SecretStr, ValidationError, field_validator
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

    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
        description="OpenAI API 密钥",
    )

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def normalize_optional_secret(cls, value: object) -> object:
        """把空字符串视为未配置密钥。"""

        if isinstance(value, str) and not value.strip():
            return None
        return value

    @classmethod
    def from_env(cls) -> "EnvSettings":
        """从环境变量读取配置并转换为项目统一错误。"""

        try:
            return cls()
        except ValidationError as exc:
            raise AppError(
                ErrorCode.INVALID_CONFIG,
                format_validation_error(exc),
            ) from exc
