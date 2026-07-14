"""Embedding 客户端抽象。"""

from __future__ import annotations

from typing import Protocol


class EmbeddingClient(Protocol):
    """索引流程依赖的 embedding 服务协议。"""

    @property
    def provider(self) -> str:
        """返回 embedding 服务提供方名称。"""

    @property
    def model_name(self) -> str:
        """返回当前模型名称。"""

    @property
    def dimension(self) -> int:
        """返回 embedding 向量维度。"""

    def embed_text(self, text: str) -> list[float]:
        """将单段文本转换为向量。"""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量将文本转换为向量。"""
