"""检索器基础协议。"""

from __future__ import annotations

from typing import Protocol

from app.core.models import RetrievedChunk


class Retriever(Protocol):
    """检索器协议。

    不管底层是向量检索、BM25 还是后续 hybrid/rerank，都应该返回统一的 RetrievedChunk。
    """

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """根据 query 返回相关 chunk。"""
