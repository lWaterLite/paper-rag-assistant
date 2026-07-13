"""模型上下文 token 估算接口。"""

from __future__ import annotations

from typing import Protocol


class TokenEstimator(Protocol):
    """估算文本进入生成模型后占用的 token 数量。"""

    @property
    def name(self) -> str:
        """返回稳定策略名称。"""

    def count_text(self, text: str) -> int:
        """返回文本的 token 估算值。"""
