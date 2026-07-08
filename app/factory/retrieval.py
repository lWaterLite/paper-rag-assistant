"""Retrieval 相关对象组装。"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.errors import AppError, ErrorCode
from app.factory.configs import ConfigFactory
from app.indexing.index_builder import RagIndex
from app.retrieval.configs import BM25Config
from app.retrieval.retrievers import BM25Retriever, Retriever, VectorRetriever
from app.retrieval.service import SearchService


@dataclass(slots=True)
class RetrievalFactory:
    """组装在线检索相关对象。"""

    configs: ConfigFactory

    @staticmethod
    def build_vector_retriever(index: RagIndex) -> VectorRetriever:
        """创建向量检索器。"""

        return VectorRetriever(
            index.embedding_client,
            index.vector_collection,
            index.chunk_collection,
        )

    def build_bm25_retriever(self, index: RagIndex) -> BM25Retriever:
        """根据当前 chunk collection 创建 BM25 检索器。"""

        config = BM25Config(
            k1=self.configs.project_settings.retrieval.bm25_k1,
            b=self.configs.project_settings.retrieval.bm25_b,
        )
        return BM25Retriever(index.chunk_collection.iter_chunks(), config=config)

    def build_retriever(self, index: RagIndex) -> Retriever:
        """根据检索策略创建在线问答默认检索器。"""

        retrieval_config = self.configs.build_retrieval_config()
        if retrieval_config.strategy == "vector":
            return self.build_vector_retriever(index)
        if retrieval_config.strategy == "bm25":
            return self.build_bm25_retriever(index)
        raise AppError(
            ErrorCode.INVALID_CONFIG,
            "hybrid 检索将在后续子模块实现，当前请使用 vector 或 bm25",
        )

    def build_search_service(self, index: RagIndex) -> SearchService:
        """创建只执行检索的 SearchService。"""

        return SearchService(
            retrievers={
                "vector": self.build_vector_retriever(index),
                "bm25": self.build_bm25_retriever(index),
            },
            config=self.configs.build_retrieval_config(),
        )
