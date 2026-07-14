"""候选证据变换协议。"""

from __future__ import annotations

from typing import Protocol

from app.retrieval.context.evidence_transformers.models import (
    EvidenceTransformRequest,
    EvidenceTransformResult,
)


class EvidenceTransformer(Protocol):
    """将已完成检索排序的候选转换为可追溯证据。"""

    @property
    def name(self) -> str:
        """返回稳定的 transformer 名称。"""

    def transform(self, request: EvidenceTransformRequest) -> EvidenceTransformResult:
        """变换候选证据，不负责重新检索或改变候选排序。"""
