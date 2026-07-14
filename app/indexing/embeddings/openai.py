"""OpenAI embedding 客户端。"""

from __future__ import annotations

from typing import Any

from app.core.errors import AppError, ErrorCode
from app.indexing.configuration import EmbeddingConfig
from app.indexing.embeddings.validation import validate_embedding_vectors


class OpenAIEmbeddingClient:
    """通过 OpenAI SDK 调用真实 embedding 服务。"""

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
        self._client: Any = OpenAI(
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
        """按配置批大小请求并校验所有 embedding。"""

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
                    response.data,
                    key=lambda item: getattr(item, "index", 0),
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


def _batched(items: list[str], batch_size: int) -> list[list[str]]:
    """按固定大小切分批次。"""

    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]
