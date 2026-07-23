"""LLM Client 的运行期配置。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LlmClientConfig:
    """构建 LLM Client 所需的非敏感运行期配置。"""

    provider: str
    model: str
    base_url: str | None
    timeout_seconds: float
    max_retries: int

    def __post_init__(self) -> None:
        provider = self.provider.strip()
        model = self.model.strip()
        if not provider:
            raise ValueError("LLM provider 不能为空")
        if not model:
            raise ValueError("LLM model 不能为空")
        if self.base_url is not None and not self.base_url.strip():
            object.__setattr__(self, "base_url", None)
        if self.timeout_seconds <= 0:
            raise ValueError("LLM timeout_seconds 必须大于 0")
        if self.max_retries < 0:
            raise ValueError("LLM max_retries 不能小于 0")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model", model)

