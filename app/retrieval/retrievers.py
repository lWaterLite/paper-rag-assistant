"""检索器。"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from typing import Protocol

from app.core.errors import AppError, ErrorCode
from app.core.models import DocumentChunk, RetrievedChunk
from app.indexing.embeddings import EmbeddingClient
from app.indexing.vector_collection import VectorCollection
from app.ingest.chunking.collection import ChunkCollection


class Retriever(Protocol):
    """检索器协议。

    不管底层是向量检索、BM25 还是混合检索，都应该返回统一的 RetrievedChunk。
    """

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """根据 query 返回相关 chunk。"""


class VectorRetriever:
    """基于向量相似度的检索器。"""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        vector_collection: VectorCollection,
        chunk_collection: ChunkCollection,
    ) -> None:
        self._embedding_client = embedding_client
        self._vector_collection = vector_collection
        self._chunk_collection = chunk_collection

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """检索与 query 最相似的 chunk。"""

        query_vector = self._embedding_client.embed_text(query)
        vector_results = self._vector_collection.search(query_vector, top_k=top_k)
        retrieved_chunks: list[RetrievedChunk] = []

        for result in vector_results:
            chunk = self._chunk_collection.get_by_id(result.chunk_id)
            if chunk is None:
                raise AppError(
                    ErrorCode.RETRIEVAL_FAILED,
                    f"向量命中了 chunk_id={result.chunk_id}，但 chunk 集合中找不到对应内容",
                )
            retrieved_chunks.append(_build_retrieved_chunk(chunk, score=result.score, rank=result.rank))

        return retrieved_chunks


def _build_retrieved_chunk(chunk: DocumentChunk, *, score: float, rank: int) -> RetrievedChunk:
    """把 DocumentChunk 和向量检索分数组装成 RetrievedChunk。"""

    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        content_hash=chunk.content_hash,
        version_id=chunk.version_id,
        text=chunk.text,
        score=score,
        rank=rank,
        retriever="vector",
        source_path=chunk.source_path,
        chunk_index=chunk.chunk_index,
        title=chunk.title,
        section=chunk.section,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        metadata=chunk.metadata,
    )


class BM25Retriever:
    """基于关键词匹配的 BM25 检索器。

    当前实现是教学版，分词逻辑比较简单：
    - 英文、数字、下划线按连续词提取。
    - 中文按单字提取。
    后续如果要做中文论文或中文知识库，应接入更合适的分词器。
    """

    def __init__(
        self,
        chunks: Iterable[DocumentChunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._chunks = list(chunks)
        self._k1 = k1
        self._b = b
        self._tokenized_chunks = [_tokenize(chunk.text) for chunk in self._chunks]
        self._term_frequencies = [Counter(tokens) for tokens in self._tokenized_chunks]
        self._document_frequencies = self._build_document_frequencies(self._tokenized_chunks)
        self._average_document_length = self._calculate_average_document_length(self._tokenized_chunks)

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """检索与 query 关键词最相关的 chunk。"""

        if top_k <= 0 or not self._chunks:
            return []

        query_terms = _tokenize(query)
        if not query_terms:
            return []

        scored = [
            (self._score(query_terms, index), chunk)
            for index, chunk in enumerate(self._chunks)
        ]
        scored = [(score, chunk) for score, chunk in scored if score > 0]
        scored.sort(key=lambda item: item[0], reverse=True)

        return [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                content_hash=chunk.content_hash,
                version_id=chunk.version_id,
                text=chunk.text,
                score=round(score, 4),
                rank=rank,
                retriever="bm25",
                source_path=chunk.source_path,
                chunk_index=chunk.chunk_index,
                title=chunk.title,
                section=chunk.section,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                metadata=chunk.metadata,
            )
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]

    def _score(self, query_terms: list[str], chunk_index: int) -> float:
        """计算单个 chunk 的 BM25 分数。"""

        term_frequency = self._term_frequencies[chunk_index]
        document_length = len(self._tokenized_chunks[chunk_index])
        score = 0.0

        for term in query_terms:
            frequency = term_frequency.get(term, 0)
            if frequency == 0:
                continue

            inverse_document_frequency = self._inverse_document_frequency(term)
            denominator = frequency + self._k1 * (
                1 - self._b + self._b * document_length / self._average_document_length
            )
            score += inverse_document_frequency * (frequency * (self._k1 + 1)) / denominator

        return score

    def _inverse_document_frequency(self, term: str) -> float:
        """计算 BM25 IDF。

        这里使用常见的 Okapi BM25 平滑形式，避免极端情况下出现负数或除零。
        """

        document_count = len(self._chunks)
        document_frequency = self._document_frequencies.get(term, 0)
        return math.log(1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))

    def _build_document_frequencies(self, tokenized_chunks: list[list[str]]) -> dict[str, int]:
        """统计每个词出现在多少个 chunk 中。"""

        frequencies: dict[str, int] = {}
        for tokens in tokenized_chunks:
            for token in set(tokens):
                frequencies[token] = frequencies.get(token, 0) + 1
        return frequencies

    def _calculate_average_document_length(self, tokenized_chunks: list[list[str]]) -> float:
        """计算平均文档长度。"""

        if not tokenized_chunks:
            return 1.0
        total_length = sum(len(tokens) for tokens in tokenized_chunks)
        return total_length / len(tokenized_chunks) or 1.0


def _tokenize(text: str) -> list[str]:
    """教学版 tokenizer。

    英文词按单词切分，中文按单字切分。
    """

    return re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", text.lower())
