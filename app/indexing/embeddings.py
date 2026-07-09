"""Embedding 客户端。

索引构建流程只依赖 EmbeddingClient 协议，不直接依赖某个 provider SDK。
这样 mock、本地模型和真实云端 embedding 都可以在 factory 层替换。
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

from app.core.errors import AppError, ErrorCode
from app.indexing.configs import EmbeddingConfig


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
        vector = _build_hash_vector(
            text=text,
            model_name=self.model_name,
            dimension=self.dimension,
        )
        return _normalize(vector)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


class OpenAIEmbeddingClient:
    """OpenAI embedding 客户端。

    这个类使用懒导入，只有当配置选择 openai provider 时才需要安装 openai SDK。
    API key 由组合根从 EnvSettings 显式注入，客户端不自行读取全局环境。
    """

    def __init__(self, config: EmbeddingConfig, *, api_key: str | None = None) -> None:
        self._config = config
        if not api_key:
            raise AppError(
                ErrorCode.INVALID_CONFIG,
                "缺少 OpenAI embedding API key，请在 .env 中配置 OPENAI_API_KEY",
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise AppError(
                ErrorCode.INVALID_CONFIG,
                "当前环境未安装 openai SDK。请在需要真实 OpenAI embedding 时自行添加依赖：openai",
            ) from exc

        self._client = OpenAI(
            api_key=api_key,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def dimension(self) -> int:
        return self._config.dimension

    def embed_text(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for batch in _batched(texts, self._config.batch_size):
            response = self._client.embeddings.create(
                model=self._config.model,
                input=batch,
                encoding_format="float",
                dimensions=self._config.dimension,
            )
            batch_vectors = [
                item.embedding
                for item in sorted(
                    response.data, key=lambda item: getattr(item, "index", 0)
                )
            ]
            validate_embedding_vectors(
                expected_count=len(batch),
                vectors=batch_vectors,
                expected_dimension=self.dimension,
                context="OpenAI embedding 返回结果",
            )
            vectors.extend(batch_vectors)

        validate_embedding_vectors(
            expected_count=len(texts),
            vectors=vectors,
            expected_dimension=self.dimension,
            context="OpenAI embedding 批量结果",
        )
        return vectors


def validate_embedding_vectors(
    *,
    expected_count: int,
    vectors: list[list[float]],
    expected_dimension: int,
    context: str,
) -> None:
    """校验 embedding 返回数量、维度和数值合法性。"""

    if len(vectors) != expected_count:
        raise AppError(
            ErrorCode.INDEX_FAILED,
            f"{context}数量不一致：期望 {expected_count} 个向量，实际 {len(vectors)} 个",
        )

    for index, vector in enumerate(vectors):
        if len(vector) != expected_dimension:
            raise AppError(
                ErrorCode.INDEX_FAILED,
                f"{context}维度不一致：第 {index} 个向量为 {len(vector)} 维，期望 {expected_dimension} 维",
            )
        if any(not math.isfinite(value) for value in vector):
            raise AppError(
                ErrorCode.INDEX_FAILED,
                f"{context}包含非法数值：第 {index} 个向量存在 NaN 或 Infinity",
            )


def _build_hash_vector(*, text: str, model_name: str, dimension: int) -> list[float]:
    """根据文本和模型名生成稳定 hash 向量。

    blake2b 单次 digest 最多 64 字节，所以这里按轮次扩展，支持任意正整数维度。
    """

    values: list[float] = []
    round_index = 0
    while len(values) < dimension:
        payload = f"{model_name}|{round_index}|{text}".encode("utf-8")
        remaining = dimension - len(values)
        digest = hashlib.blake2b(payload, digest_size=min(64, remaining)).digest()
        values.extend((byte - 128) / 128 for byte in digest)
        round_index += 1
    return values


def _normalize(vector: list[float]) -> list[float]:
    """把向量归一化，方便余弦相似度检索。"""

    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _batched(items: list[str], batch_size: int) -> list[list[str]]:
    """按固定大小切分 batch。"""

    return [
        items[index : index + batch_size] for index in range(0, len(items), batch_size)
    ]
