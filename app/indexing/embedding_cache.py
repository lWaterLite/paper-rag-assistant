"""Embedding 缓存。

真实 RAG 项目中，embedding 通常是离线索引阶段最贵的步骤之一。
缓存的目标是：同一个 embedding 模型下，同一段文本不要重复计算向量。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from app.indexing.embeddings import EmbeddingClient


@dataclass(frozen=True)
class EmbeddingCacheKey:
    """Embedding 缓存键。

    key 同时包含模型信息和文本 hash。
    这样换 provider、换模型、换维度后，不会错误复用旧 embedding。
    """

    provider: str
    model_name: str
    dimension: int
    text_hash: str


class EmbeddingCache(Protocol):
    """Embedding 缓存协议。"""

    def get(self, client: EmbeddingClient, text: str) -> list[float] | None:
        """读取缓存。"""

    def set(self, client: EmbeddingClient, text: str, vector: list[float]) -> None:
        """写入缓存。"""

    def count(self) -> int:
        """返回缓存条目数量。"""


class InMemoryEmbeddingCache:
    """内存 embedding 缓存。

    当前只服务于练习和测试。后续可以替换成 JSONL、SQLite、Redis 或对象存储。
    """

    def __init__(self) -> None:
        self._items: dict[EmbeddingCacheKey, list[float]] = {}

    def get(self, client: EmbeddingClient, text: str) -> list[float] | None:
        return self._items.get(_build_cache_key(client, text))

    def set(self, client: EmbeddingClient, text: str, vector: list[float]) -> None:
        self._items[_build_cache_key(client, text)] = vector

    def count(self) -> int:
        return len(self._items)


def _build_cache_key(client: EmbeddingClient, text: str) -> EmbeddingCacheKey:
    """根据 embedding 模型信息和文本生成缓存键。"""

    return EmbeddingCacheKey(
        provider=client.provider,
        model_name=client.model_name,
        dimension=client.dimension,
        text_hash=_hash_text(text),
    )


def _hash_text(text: str) -> str:
    """生成文本 hash。"""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()

