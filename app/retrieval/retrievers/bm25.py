"""BM25 关键词检索。"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from app.core.models import DocumentChunk, RetrievedChunk
from app.retrieval.configs import BM25Config
from app.retrieval.retrievers.result_builder import RetrievedChunkBuilder


@dataclass(frozen=True)
class BM25SearchHit:
    """BM25 索引内部命中结果。"""

    chunk: DocumentChunk
    score: float
    rank: int


class BM25Index:
    """基于 DocumentChunk 构建的内存 BM25 索引。

    这个类负责维护关键词检索需要的统计信息，BM25Retriever 只负责把搜索命中转换成统一结果。
    """

    def __init__(self, chunks: Iterable[DocumentChunk], config: BM25Config) -> None:
        self._chunks = list(chunks)
        self._config = config
        self._tokenized_chunks = [tokenize_basic(chunk.text) for chunk in self._chunks]
        self._term_frequencies = [Counter(tokens) for tokens in self._tokenized_chunks]
        self._document_frequencies = self._build_document_frequencies(
            self._tokenized_chunks
        )
        self._average_document_length = self._calculate_average_document_length(
            self._tokenized_chunks
        )

    @classmethod
    def from_chunks(
        cls,
        chunks: Iterable[DocumentChunk],
        *,
        config: BM25Config | None = None,
    ) -> "BM25Index":
        """根据 chunks 创建 BM25 索引。"""

        return cls(chunks, config or BM25Config())

    @property
    def chunk_count(self) -> int:
        """索引中的 chunk 数量。"""

        return len(self._chunks)

    @property
    def average_document_length(self) -> float:
        """BM25 使用的平均文档长度。"""

        return self._average_document_length

    def search(self, query: str, top_k: int) -> list[BM25SearchHit]:
        """搜索与 query 关键词最相关的 chunks。"""

        if top_k <= 0 or not self._chunks:
            return []

        query_terms = tokenize_basic(query)
        if not query_terms:
            return []

        scored = [
            (self._score(query_terms, index), chunk)
            for index, chunk in enumerate(self._chunks)
        ]
        scored = [(score, chunk) for score, chunk in scored if score > 0]
        scored.sort(key=lambda item: item[0], reverse=True)

        return [
            BM25SearchHit(chunk=chunk, score=score, rank=rank)
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
            denominator = frequency + self._config.k1 * (
                1
                - self._config.b
                + self._config.b * document_length / self._average_document_length
            )
            score += (
                inverse_document_frequency
                * (frequency * (self._config.k1 + 1))
                / denominator
            )

        return score

    def _inverse_document_frequency(self, term: str) -> float:
        """计算 BM25 IDF。"""

        document_count = len(self._chunks)
        document_frequency = self._document_frequencies.get(term, 0)
        return math.log(
            1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )

    @staticmethod
    def _build_document_frequencies(
        tokenized_chunks: list[list[str]],
    ) -> dict[str, int]:
        """统计每个词出现在多少个 chunk 中。"""

        frequencies: dict[str, int] = {}
        for tokens in tokenized_chunks:
            for token in set(tokens):
                frequencies[token] = frequencies.get(token, 0) + 1
        return frequencies

    @staticmethod
    def _calculate_average_document_length(tokenized_chunks: list[list[str]]) -> float:
        """计算平均文档长度。"""

        if not tokenized_chunks:
            return 1.0
        total_length = sum(len(tokens) for tokens in tokenized_chunks)
        return total_length / len(tokenized_chunks) or 1.0


class BM25Retriever:
    """基于 BM25Index 的关键词检索器。"""

    def __init__(
        self,
        chunks_or_index: Iterable[DocumentChunk] | BM25Index,
        *,
        config: BM25Config | None = None,
        k1: float | None = None,
        b: float | None = None,
    ) -> None:
        resolved_config = config or BM25Config(
            k1=BM25Config().k1 if k1 is None else k1,
            b=BM25Config().b if b is None else b,
        )
        if isinstance(chunks_or_index, BM25Index):
            self._index = chunks_or_index
        else:
            self._index = BM25Index.from_chunks(chunks_or_index, config=resolved_config)

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """检索与 query 关键词最相关的 chunk。"""

        return [
            RetrievedChunkBuilder.from_chunk(
                hit.chunk,
                score=round(hit.score, 4),
                rank=hit.rank,
                retriever="bm25",
            )
            for hit in self._index.search(query, top_k=top_k)
        ]


def tokenize_basic(text: str) -> list[str]:
    """教学版 tokenizer。

    英文词按单词切分，中文按单字切分。后续可以替换为专业 tokenizer。
    """

    return re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", text.lower())
