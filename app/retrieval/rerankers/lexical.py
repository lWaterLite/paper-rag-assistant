"""无需外部模型的确定性 lexical reranker。"""

from __future__ import annotations

import math
from collections.abc import Sequence

from app.retrieval.models import RetrievedChunk
from app.retrieval.rerankers.base import RerankedCandidate
from app.retrieval.tokenizers.base import Tokenizer


class LexicalReranker:
    """根据 query token 覆盖率和短语命中重排候选。

    这是用于验证 rerank 结构、配置和报告链路的确定性 baseline，
    并不试图替代真实 cross-encoder 的语义判别能力。
    """

    def __init__(self, tokenizer: Tokenizer, *, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("lexical reranker batch_size 必须大于 0")
        self._tokenizer = tokenizer
        self._batch_size = batch_size

    @property
    def name(self) -> str:
        """返回稳定的策略名称。"""

        return "lexical"

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        limit: int,
    ) -> list[RerankedCandidate]:
        """对候选打分并以原始 rank 作为稳定的并列排序规则。"""

        if limit <= 0:
            return []

        query_tokens = frozenset(self._tokenizer.tokenize(query))
        normalized_query = " ".join(query.lower().split())
        scored_candidates: list[RerankedCandidate] = []
        for start in range(0, len(candidates), self._batch_size):
            batch = candidates[start : start + self._batch_size]
            scored_candidates.extend(
                RerankedCandidate(
                    chunk=chunk,
                    score=self._score_chunk(
                        chunk,
                        query_tokens=query_tokens,
                        normalized_query=normalized_query,
                    ),
                )
                for chunk in batch
            )
        scored_candidates.sort(
            key=lambda item: (-item.score, item.chunk.rank, item.chunk.chunk_id)
        )
        return scored_candidates[:limit]

    def _score_chunk(
        self,
        chunk: RetrievedChunk,
        *,
        query_tokens: frozenset[str],
        normalized_query: str,
    ) -> float:
        """计算稳定、可解释的 lexical 相关性分数。"""

        if not query_tokens:
            return 0.0

        chunk_tokens = self._tokenizer.tokenize(chunk.text)
        if not chunk_tokens:
            return 0.0

        overlap_count = len(query_tokens.intersection(chunk_tokens))
        coverage = overlap_count / len(query_tokens)
        density = overlap_count / math.sqrt(len(chunk_tokens))
        normalized_text = " ".join(chunk.text.lower().split())
        phrase_bonus = 1.0 if normalized_query and normalized_query in normalized_text else 0.0
        return round(coverage * 2.0 + density + phrase_bonus, 6)
