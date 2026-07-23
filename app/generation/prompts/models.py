"""Prompt 构建结果模型。"""

from __future__ import annotations

from dataclasses import dataclass

from app.llm import LlmMessage


@dataclass(frozen=True, slots=True)
class RagAnswerPrompt:
    """一次 RAG 回答生成所需的版本化 prompt。"""

    version: str
    system_prompt: str
    user_prompt: str

    @property
    def messages(self) -> tuple[LlmMessage, ...]:
        """转换为 LLM Client 使用的消息列表。"""

        return (
            LlmMessage(role="system", content=self.system_prompt),
            LlmMessage(role="user", content=self.user_prompt),
        )

