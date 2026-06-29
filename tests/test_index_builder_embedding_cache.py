"""IndexBuilder 的 embedding cache 测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

from app.core.config import Settings
from app.factory import build_index_builder
from app.indexing.embedding_cache import InMemoryEmbeddingCache


class CountingEmbeddingClient:
    """记录实际 embedding 调用文本数量的测试 client。"""

    def __init__(self, dimension: int = 2) -> None:
        self._dimension = dimension
        self.embedded_text_count = 0

    @property
    def provider(self) -> str:
        return "test"

    @property
    def model_name(self) -> str:
        return "counting-embedding"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        self.embedded_text_count += 1
        return [float(len(text)), 1.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.embedded_text_count += len(texts)
        return [[float(len(text)), 1.0] for text in texts]


class IndexBuilderEmbeddingCacheTest(unittest.TestCase):
    """验证 IndexBuilder 会复用 embedding cache。"""

    def test_second_build_reuses_embedding_cache(self) -> None:
        settings = Settings(chunk_size=120, chunk_overlap=20)
        cache = InMemoryEmbeddingCache()
        client = CountingEmbeddingClient()

        first_builder = build_index_builder(settings, embedding_client=client, embedding_cache=cache)
        _, first_result = first_builder.build_from_directory(Path("data/raw/papers"))

        second_builder = build_index_builder(settings, embedding_client=client, embedding_cache=cache)
        _, second_result = second_builder.build_from_directory(Path("data/raw/papers"))

        self.assertEqual(first_result.embedding_cache_hits, 0)
        self.assertEqual(first_result.embedding_cache_misses, first_result.chunk_count)
        self.assertEqual(second_result.embedding_cache_hits, second_result.chunk_count)
        self.assertEqual(second_result.embedding_cache_misses, 0)
        self.assertEqual(client.embedded_text_count, first_result.chunk_count)

    def test_same_builder_skips_chunks_already_in_vector_store(self) -> None:
        settings = Settings(chunk_size=120, chunk_overlap=20)
        client = CountingEmbeddingClient()
        builder = build_index_builder(settings, embedding_client=client)

        _, first_result = builder.build_from_directory(Path("data/raw/papers"))
        index, second_result = builder.build_from_directory(Path("data/raw/papers"))

        self.assertEqual(second_result.skipped_existing_chunks, first_result.chunk_count)
        self.assertEqual(second_result.embedding_cache_hits, 0)
        self.assertEqual(second_result.embedding_cache_misses, 0)
        self.assertEqual(index.vector_store.count(), first_result.chunk_count)
        self.assertEqual(client.embedded_text_count, first_result.chunk_count)


if __name__ == "__main__":
    unittest.main()
