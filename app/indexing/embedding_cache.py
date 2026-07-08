"""Embedding 缓存。

真实 RAG 项目中，embedding 通常是离线索引阶段最贵的步骤之一。
缓存的目标是：同一个 embedding 模型下，同一段文本不要重复计算向量。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
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

    def persist(self) -> None:
        """持久化缓存。内存实现可以是空操作。"""


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

    def persist(self) -> None:
        """内存缓存不需要持久化。"""


class FileEmbeddingCache:
    """基于 JSON 文件的 embedding 缓存。

    当前实现面向本地学习和中小规模索引构建。真实大规模场景可以替换为 SQLite、Redis 或数据库表。
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._items: dict[EmbeddingCacheKey, list[float]] = {}
        self._load_if_exists()

    def get(self, client: EmbeddingClient, text: str) -> list[float] | None:
        return self._items.get(_build_cache_key(client, text))

    def set(self, client: EmbeddingClient, text: str, vector: list[float]) -> None:
        self._items[_build_cache_key(client, text)] = vector

    def count(self) -> int:
        return len(self._items)

    def persist(self) -> None:
        """把缓存写入 JSON 文件。

        目录创建由 IndexBuilder 在流程准备阶段完成，这里只负责写文件。
        """

        payload = [
            {
                "provider": key.provider,
                "model_name": key.model_name,
                "dimension": key.dimension,
                "text_hash": key.text_hash,
                "vector": vector,
            }
            for key, vector in sorted(
                self._items.items(), key=lambda item: _cache_key_to_string(item[0])
            )
        ]
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _load_if_exists(self) -> None:
        """加载已有缓存文件。"""

        if not self._path.exists():
            return

        data = json.loads(self._path.read_text(encoding="utf-8"))
        for item in data:
            key = EmbeddingCacheKey(
                provider=str(item["provider"]),
                model_name=str(item["model_name"]),
                dimension=int(item["dimension"]),
                text_hash=str(item["text_hash"]),
            )
            self._items[key] = [float(value) for value in item["vector"]]


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


def _cache_key_to_string(key: EmbeddingCacheKey) -> str:
    """把缓存键转换为稳定排序字符串。"""

    return f"{key.provider}|{key.model_name}|{key.dimension}|{key.text_hash}"
