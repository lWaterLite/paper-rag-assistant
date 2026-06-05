"""Embedding 客户端。

当前使用 mock embedding，保证不依赖真实外部服务。
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

from app.core.config import Settings


class EmbeddingClient(Protocol):
    """Embedding 客户端协议。"""

    @property
    def provider(self) -> str:
        """Embedding 服务提供方。"""

    @property
    def model_name(self) -> str:
        """Embedding 模型名称。"""

    @property
    def dimension(self) -> int:
        """Embedding 向量维度。"""

    def embed_text(self, text: str) -> list[float]:
        """将单段文本转换为向量。"""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量将文本转换为向量。"""


class MockEmbeddingClient:
    """可复现的 mock embedding。

    它不理解真实语义，只适合学习 pipeline 结构和编写测试。
    """

    def __init__(self, settings: Settings) -> None:
        self._dimension = settings.mock_embedding_dimension

    @property
    def provider(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-hash-embedding"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=self._dimension).digest()
        vector = [(byte - 128) / 128 for byte in digest]
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    # TODO 练习 5：
    # 当前 mock embedding 不具备真实语义能力。
    # 请你在 README 中说明它的局限，并思考真实 EmbeddingClient 需要哪些配置：
    # 1. provider
    # 2. model
    # 3. api_key
    # 4. batch_size
    # 5. timeout
