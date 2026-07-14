"""ContextPacker 前的可追溯候选证据变换能力。"""

from app.retrieval.context.evidence_transformers.base import EvidenceTransformer
from app.retrieval.context.evidence_transformers.config import (
    EvidenceTransformationConfig,
)
from app.retrieval.context.evidence_transformers.models import (
    EvidenceCandidate,
    EvidenceSource,
    EvidenceTransformRequest,
    EvidenceTransformResult,
    EvidenceTransformStageResult,
    passthrough_candidates,
)
from app.retrieval.context.evidence_transformers.passthrough import (
    PassthroughEvidenceTransformer,
)
from app.retrieval.context.evidence_transformers.registry import (
    EvidenceTransformerRegistry,
    build_default_evidence_transformer_registry,
)
from app.retrieval.context.evidence_transformers.stage import EvidenceTransformStage

__all__ = [
    "EvidenceCandidate",
    "EvidenceSource",
    "EvidenceTransformRequest",
    "EvidenceTransformResult",
    "EvidenceTransformStage",
    "EvidenceTransformStageResult",
    "EvidenceTransformationConfig",
    "EvidenceTransformer",
    "EvidenceTransformerRegistry",
    "PassthroughEvidenceTransformer",
    "build_default_evidence_transformer_registry",
    "passthrough_candidates",
]
