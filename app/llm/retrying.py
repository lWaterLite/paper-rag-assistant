"""LLM Client 的有限重试装饰器。"""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.llm.base import LlmClient
from app.llm.models import LlmRequest, LlmResponse
from app.llm.openai_compatible import LlmClientRequestError


@dataclass(frozen=True, slots=True)
class RetryingLlmClient:
    """仅对明确标记为可重试的基础设施错误执行有限退避。"""

    delegate: LlmClient
    max_retries: int
    base_delay_seconds: float = 0.2

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries 不能小于 0")
        if self.base_delay_seconds <= 0:
            raise ValueError("base_delay_seconds 必须大于 0")

    @property
    def provider_name(self) -> str:
        """保持底层 provider 名称稳定。"""

        return self.delegate.provider_name

    def complete(self, request: LlmRequest) -> LlmResponse:
        """调用底层 Client，并在短暂服务故障时有限重试。"""

        for attempt in range(self.max_retries + 1):
            try:
                return self.delegate.complete(request)
            except LlmClientRequestError as exc:
                if not exc.retriable or attempt >= self.max_retries:
                    raise
                time.sleep(self.base_delay_seconds * (2**attempt))
        raise RuntimeError("LLM 重试流程内部错误")

