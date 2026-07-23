"""生成请求的实际 token 总预算校验。"""

from __future__ import annotations

from dataclasses import dataclass

from app.generation.prompts import RagAnswerPrompt
from app.retrieval.context.packer import ContextPackerConfig
from app.retrieval.context.token_estimators.base import TokenEstimator


@dataclass(frozen=True, slots=True)
class PromptBudgetUsage:
    """一次生成请求在模型窗口内的 token 预算明细。"""

    prompt_tokens: int
    max_output_tokens: int
    safety_margin_tokens: int
    total_reserved_tokens: int
    model_context_window: int


@dataclass(frozen=True, slots=True)
class PromptBudgetValidator:
    """在调用模型前校验完整 prompt 与输出预留不会超窗。"""

    context_config: ContextPackerConfig
    token_estimator: TokenEstimator

    def validate(
        self,
        prompt: RagAnswerPrompt,
        *,
        max_output_tokens: int,
    ) -> PromptBudgetUsage:
        """计算完整消息 token，并在可能超窗时拒绝请求。"""

        prompt_tokens = sum(
            self.token_estimator.count_text(message.content)
            for message in prompt.messages
        )
        total_reserved_tokens = (
            prompt_tokens
            + max_output_tokens
            + self.context_config.safety_margin_tokens
        )
        if total_reserved_tokens > self.context_config.model_context_window:
            raise ValueError(
                "生成请求 token 预算超过模型窗口："
                f"prompt={prompt_tokens}，output={max_output_tokens}，"
                f"safety_margin={self.context_config.safety_margin_tokens}，"
                f"window={self.context_config.model_context_window}"
            )
        return PromptBudgetUsage(
            prompt_tokens=prompt_tokens,
            max_output_tokens=max_output_tokens,
            safety_margin_tokens=self.context_config.safety_margin_tokens,
            total_reserved_tokens=total_reserved_tokens,
            model_context_window=self.context_config.model_context_window,
        )

