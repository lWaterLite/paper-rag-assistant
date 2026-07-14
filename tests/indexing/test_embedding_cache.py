"""Embedding 缓存测试。"""

from __future__ import annotations

import unittest
import shutil
import uuid
from pathlib import Path

from app.indexing.embeddings import FileEmbeddingCache, InMemoryEmbeddingCache


class FakeEmbeddingClient:
    """测试用 embedding client。"""

    def __init__(self, *, model_name: str = "fake", dimension: int = 2) -> None:
        self._model_name = model_name
        self._dimension = dimension

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        return [float(len(text)), 1.0][: self._dimension]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


class EmbeddingCacheTest(unittest.TestCase):
    """验证 embedding cache key 的隔离行为。"""

    def test_cache_reuses_same_text_with_same_model(self) -> None:
        cache = InMemoryEmbeddingCache()
        client = FakeEmbeddingClient(model_name="model-a", dimension=2)

        cache.set(client, "same text", [1.0, 0.0])

        self.assertEqual(cache.get(client, "same text"), [1.0, 0.0])
        self.assertEqual(cache.count(), 1)

    def test_cache_is_isolated_by_model_name(self) -> None:
        cache = InMemoryEmbeddingCache()
        first_client = FakeEmbeddingClient(model_name="model-a", dimension=2)
        second_client = FakeEmbeddingClient(model_name="model-b", dimension=2)

        cache.set(first_client, "same text", [1.0, 0.0])

        self.assertIsNone(cache.get(second_client, "same text"))

    def test_cache_is_isolated_by_dimension(self) -> None:
        cache = InMemoryEmbeddingCache()
        first_client = FakeEmbeddingClient(model_name="model-a", dimension=2)
        second_client = FakeEmbeddingClient(model_name="model-a", dimension=3)

        cache.set(first_client, "same text", [1.0, 0.0])

        self.assertIsNone(cache.get(second_client, "same text"))

    def test_file_cache_can_be_reloaded(self) -> None:
        cache_dir = Path(".tmp_tests") / f"embedding_cache_{uuid.uuid4().hex}"
        cache_path = cache_dir / "embedding_cache.json"
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            client = FakeEmbeddingClient(model_name="model-a", dimension=2)
            cache = FileEmbeddingCache(cache_path)
            cache.set(client, "same text", [1.0, 0.0])
            cache.persist()

            loaded_cache = FileEmbeddingCache(cache_path)

            self.assertEqual(loaded_cache.get(client, "same text"), [1.0, 0.0])
            self.assertEqual(loaded_cache.count(), 1)
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
