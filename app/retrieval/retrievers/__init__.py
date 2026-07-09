"""检索器实现集合。

这个子包只放“检索器协议与具体检索器实现”。
检索服务编排、去重、上下文组织、结果组装等能力仍然保留在上一级 `app.retrieval` 包中。
"""

from __future__ import annotations

from app.retrieval.retrievers.base import Retriever
from app.retrieval.retrievers.bm25 import (
    BM25Index,
    BM25Retriever,
    BM25SearchHit,
)
from app.retrieval.retrievers.hybrid import HybridRetrievalSource, HybridRetriever
from app.retrieval.retrievers.result_builder import RetrievedChunkBuilder
from app.retrieval.retrievers.registry import RetrieverRegistry
from app.retrieval.retrievers.vector import VectorRetriever

__all__ = [
    "BM25Index",
    "BM25Retriever",
    "BM25SearchHit",
    "HybridRetrievalSource",
    "HybridRetriever",
    "RetrievedChunkBuilder",
    "Retriever",
    "RetrieverRegistry",
    "VectorRetriever",
]
