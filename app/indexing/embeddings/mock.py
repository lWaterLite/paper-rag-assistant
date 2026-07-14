"""用于学习与测试的可复现 embedding 实现。"""

from __future__ import annotations

import hashlib
import math

from app.indexing.configuration import EmbeddingConfig


class MockEmbeddingClient:
    """基于文本哈希的确定性 embedding 客户端。"""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config

    @property
    def provider(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def dimension(self) -> int:
        return self._config.dimension

    def embed_text(self, text: str) -> list[float]:
        return _normalize(
            _build_hash_vector(
                text=text,
                model_name=self.model_name,
                dimension=self.dimension,
            )
        )

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


def _build_hash_vector(*, text: str, model_name: str, dimension: int) -> list[float]:
    """生成指定维度的稳定哈希向量。"""

    values: list[float] = []
    round_index = 0
    while len(values) < dimension:
        payload = f"{model_name}|{round_index}|{text}".encode("utf-8")
        digest = hashlib.blake2b(
            payload,
            digest_size=min(64, dimension - len(values)),
        ).digest()
        values.extend((byte - 128) / 128 for byte in digest)
        round_index += 1
    return values


def _normalize(vector: list[float]) -> list[float]:
    """将向量归一化，方便余弦相似度检索。"""

    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]
