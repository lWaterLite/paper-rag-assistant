"""回答生成能力协议。"""

from __future__ import annotations

from typing import Protocol

from app.core.tracing import RagTrace
from app.generation.models import RagAnswer
from app.retrieval.context import PackedContext
from app.retrieval.models import RetrievedChunk


class AnswerGenerator(Protocol):
    """基于上下文生成最终回答。"""

    def generate(
        self,
        question: str,
        packed_context: PackedContext,
        retrieved_chunks: list[RetrievedChunk],
        trace: RagTrace,
    ) -> RagAnswer:
        """返回通过 citation 契约约束后的回答。"""
