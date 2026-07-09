"""分词器基础协议。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Tokenizer(Protocol):
    """文本分词策略协议。"""

    def tokenize(self, text: str) -> Sequence[str]:
        """将文本转换为可重复读取的词元序列。"""
