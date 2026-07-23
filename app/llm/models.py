"""与具体模型供应商无关的 LLM 数据契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping


LlmRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class LlmMessage:
    """一次 LLM 请求中的单条消息。"""

    role: LlmRole
    content: str


@dataclass(frozen=True, slots=True)
class LlmRequest:
    """业务层发给 LLM Client 的稳定请求模型。"""

    messages: tuple[LlmMessage, ...]
    model: str
    temperature: float
    max_output_tokens: int
    timeout_seconds: float
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("LLM 请求至少需要一条消息")
        if not self.model.strip():
            raise ValueError("LLM model 不能为空")
        if not 0 <= self.temperature <= 2:
            raise ValueError("LLM temperature 必须在 0 到 2 之间")
        if self.max_output_tokens <= 0:
            raise ValueError("LLM max_output_tokens 必须大于 0")
        if self.timeout_seconds <= 0:
            raise ValueError("LLM timeout_seconds 必须大于 0")


@dataclass(frozen=True, slots=True)
class LlmUsage:
    """模型供应商返回的 token 用量；未知时字段为 None。"""

    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class LlmResponse:
    """由 LLM Client 归一化后的响应。"""

    content: str
    model: str
    provider_request_id: str | None = None
    finish_reason: str | None = None
    usage: LlmUsage = field(default_factory=LlmUsage)

