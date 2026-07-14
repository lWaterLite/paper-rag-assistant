"""Embedding client 测试。"""

from __future__ import annotations

import unittest

from app.core.errors import AppError, ErrorCode
from app.indexing.configuration import EmbeddingConfig
from app.indexing.embeddings import (
    MockEmbeddingClient,
    OpenAIEmbeddingClient,
    build_default_embedding_client_registry,
    validate_embedding_vectors,
)


class EmbeddingClientTest(unittest.TestCase):
    """验证 embedding client 的基础工程边界。"""

    def test_mock_embedding_supports_dimension_larger_than_blake2b_digest_limit(self) -> None:
        client = MockEmbeddingClient(EmbeddingConfig(provider="mock", model="mock-hash-embedding", dimension=128))

        vector = client.embed_text("RAG indexing")

        self.assertEqual(len(vector), 128)

    def test_mock_embedding_is_stable_for_same_text_and_model(self) -> None:
        client = MockEmbeddingClient(EmbeddingConfig(provider="mock", model="mock-hash-embedding", dimension=16))

        self.assertEqual(client.embed_text("same text"), client.embed_text("same text"))

    def test_registry_creates_configured_mock_client(self) -> None:
        registry = build_default_embedding_client_registry()

        client = registry.create(
            EmbeddingConfig(
                provider="mock",
                model="mock-hash-embedding",
                dimension=16,
            )
        )

        self.assertIsInstance(client, MockEmbeddingClient)

    def test_registry_rejects_unknown_provider(self) -> None:
        registry = build_default_embedding_client_registry()

        with self.assertRaisesRegex(ValueError, "不支持的 embedding provider"):
            registry.create(
                EmbeddingConfig(
                    provider="custom",
                    model="custom-embedding",
                    dimension=16,
                )
            )

    def test_validate_embedding_vectors_rejects_dimension_mismatch(self) -> None:
        with self.assertRaises(AppError) as context:
            validate_embedding_vectors(
                expected_count=1,
                vectors=[[1.0, 0.0]],
                expected_dimension=3,
                context="测试 embedding",
            )

        self.assertEqual(context.exception.code, ErrorCode.INDEX_FAILED)
        self.assertIn("维度不一致", context.exception.message)

    def test_openai_embedding_client_rejects_missing_api_key(self) -> None:
        config = EmbeddingConfig(provider="openai", model="text-embedding-3-small", dimension=1536)
        with self.assertRaises(AppError) as context:
            OpenAIEmbeddingClient(config)

        self.assertEqual(context.exception.code, ErrorCode.INVALID_CONFIG)
        self.assertIn("OPENAI_API_KEY", context.exception.message)


if __name__ == "__main__":
    unittest.main()
