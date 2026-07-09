"""Retrieval 相关对象组装。"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.errors import AppError, ErrorCode
from app.factory.configs import ConfigFactory
from app.indexing.index_builder import RagIndex
from app.retrieval.retrievers import (
    BM25Index,
    BM25Retriever,
    HybridRetrievalSource,
    HybridRetriever,
    Retriever,
    RetrieverRegistry,
    VectorRetriever,
)
from app.retrieval.retrievers.fusion import ReciprocalRankFusion
from app.retrieval.service import SearchService
from app.retrieval.tokenizers import (
    Tokenizer,
    TokenizerRegistry,
    build_default_tokenizer_registry,
)


@dataclass(slots=True)
class RetrievalFactory:
    """组装在线检索相关对象。"""

    configs: ConfigFactory
    tokenizer_registry: TokenizerRegistry = field(
        default_factory=build_default_tokenizer_registry
    )

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

        retrieval_config = self.configs.build_retrieval_config()
        bm25_index = BM25Index.from_chunks(
            index.chunk_collection.iter_chunks(),
            config=retrieval_config.bm25,
            tokenizer=self.build_tokenizer(),
        )
        return BM25Retriever(bm25_index)

    def build_tokenizer(self) -> Tokenizer:
        """根据当前配置创建 BM25 使用的分词器。"""

        return self.tokenizer_registry.create(
            self.configs.build_tokenizer_config()
        )

    def build_hybrid_retriever(
        self,
        *,
        vector_retriever: Retriever,
        bm25_retriever: Retriever,
    ) -> HybridRetriever:
        """使用现有的向量和 BM25 检索器创建 hybrid 检索器。"""

        config = self.configs.build_hybrid_retrieval_config()
        return HybridRetriever(
            sources=(
                HybridRetrievalSource(
                    name="vector",
                    retriever=vector_retriever,
                    weight=config.vector_weight,
                ),
                HybridRetrievalSource(
                    name="bm25",
                    retriever=bm25_retriever,
                    weight=config.bm25_weight,
                ),
            ),
            fusion_strategy=ReciprocalRankFusion(
                rank_constant=config.rrf_rank_constant
            ),
            config=config,
        )

    def build_retriever_registry(self, index: RagIndex) -> RetrieverRegistry:
        """为一个 RagIndex 创建内置检索策略的惰性注册表。"""

        registry = RetrieverRegistry()
        registry.register("vector", lambda: self.build_vector_retriever(index))
        registry.register("bm25", lambda: self.build_bm25_retriever(index))
        registry.register(
            "hybrid",
            lambda: self.build_hybrid_retriever(
                vector_retriever=registry.resolve("vector"),
                bm25_retriever=registry.resolve("bm25"),
            ),
        )
        return registry

    def build_retriever(
        self,
        index: RagIndex,
        *,
        registry: RetrieverRegistry | None = None,
    ) -> Retriever:
        """根据检索策略创建在线问答默认检索器。"""

        active_registry = registry or self.build_retriever_registry(index)
        strategy = self.configs.build_retrieval_config().strategy
        try:
            return active_registry.resolve(strategy)
        except ValueError as exc:
            raise AppError(
                ErrorCode.INVALID_CONFIG,
                str(exc),
            ) from exc

    def build_search_service(
        self,
        index: RagIndex,
        *,
        registry: RetrieverRegistry | None = None,
    ) -> SearchService:
        """创建只执行检索的 SearchService。"""

        return SearchService(
            registry=registry or self.build_retriever_registry(index),
            config=self.configs.build_retrieval_config(),
        )
