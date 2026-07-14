"""不改变文本的候选证据变换器。"""

from __future__ import annotations

from app.retrieval.context.evidence_transformers.models import (
    EvidenceTransformRequest,
    EvidenceTransformResult,
    passthrough_candidates,
)


class PassthroughEvidenceTransformer:
    """将每个 RetrievedChunk 一对一包装为 EvidenceCandidate。"""

    @property
    def name(self) -> str:
        """返回策略名称。"""

        return "passthrough"

    def transform(self, request: EvidenceTransformRequest) -> EvidenceTransformResult:
        """保留原始文本与完整来源范围，用于验证流程与契约。"""

        return EvidenceTransformResult(
            candidates=passthrough_candidates(request.chunks)
        )
