"""重排序器的领域接口与结果模型。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.retrieval.models import RetrievedChunk


@dataclass(frozen=True, slots=True)
class RerankedCandidate:
    """Reranker 对一个候选 chunk 给出的评分结果。"""

    chunk: RetrievedChunk
    score: float


class Reranker(Protocol):
    """对已召回候选进行重排序的策略接口。"""

    @property
    def name(self) -> str:
        """返回用于 trace 与报告的稳定策略名称。"""

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        limit: int,
    ) -> list[RerankedCandidate]:
        """返回按相关性降序排列的候选，不修改输入对象。"""
