"""LLM Client 能力协议。"""

from __future__ import annotations

from typing import Protocol

from app.llm.models import LlmRequest, LlmResponse


class LlmClient(Protocol):
    """执行一次与供应商无关的文本生成请求。"""

    @property
    def provider_name(self) -> str:
        """返回稳定的供应商或实现名称。"""

    def complete(self, request: LlmRequest) -> LlmResponse:
        """完成一次非流式生成。

        Streaming 属于子模块 9 的服务化边界；本阶段先保持同步、可校验的完整响应契约。
        """

