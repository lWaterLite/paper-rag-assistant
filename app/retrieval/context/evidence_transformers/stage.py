"""候选证据变换阶段。"""

from __future__ import annotations

from app.core.errors import AppError, ErrorCode
from app.retrieval.context.evidence_transformers.base import EvidenceTransformer
from app.retrieval.context.evidence_transformers.config import (
    EvidenceTransformationConfig,
)
from app.retrieval.context.evidence_transformers.models import (
    EvidenceCandidate,
    EvidenceTransformRequest,
    EvidenceTransformStageResult,
    passthrough_candidates,
)


class EvidenceTransformStage:
    """在 ContextPacker 前执行候选证据变换并验证来源契约。"""

    def __init__(
        self,
        *,
        config: EvidenceTransformationConfig,
        transformer: EvidenceTransformer | None,
    ) -> None:
        if config.enabled and transformer is None:
            raise ValueError("启用 evidence transformation 时必须显式注入 transformer")
        if not config.enabled and transformer is not None:
            raise ValueError("禁用 evidence transformation 时不应注入 transformer")
        self._config = config
        self._transformer = transformer

    def process(
        self,
        request: EvidenceTransformRequest,
    ) -> EvidenceTransformStageResult:
        """变换候选；fail-open 时回退到未经变换的完整 chunk。"""

        if not self._config.enabled:
            return EvidenceTransformStageResult(
                candidates=passthrough_candidates(request.chunks),
                detail={
                    "transformer": "disabled",
                    "failure_mode": self._config.failure_mode,
                    "degraded": False,
                },
            )

        if self._transformer is None:
            raise RuntimeError("evidence transformation 内部错误：缺少 transformer")

        try:
            result = self._transformer.transform(request)
            self._validate_result(request, result.candidates)
        except Exception as exc:
            if self._config.failure_mode == "fail_open":
                return EvidenceTransformStageResult(
                    candidates=passthrough_candidates(request.chunks),
                    detail={
                        "transformer": self._transformer.name,
                        "failure_mode": self._config.failure_mode,
                        "degraded": True,
                        "error_message": str(exc),
                    },
                )
            raise AppError(
                ErrorCode.EVIDENCE_TRANSFORM_FAILED,
                f"evidence transformer {self._transformer.name} 执行失败：{exc}",
            ) from exc

        return EvidenceTransformStageResult(
            candidates=result.candidates,
            detail={
                "transformer": self._transformer.name,
                "failure_mode": self._config.failure_mode,
                "degraded": False,
            },
        )

    @staticmethod
    def _validate_result(
        request: EvidenceTransformRequest,
        candidates: tuple[EvidenceCandidate, ...],
    ) -> None:
        """确保变换只改变证据形态，不静默改变候选覆盖范围与排序。"""

        original_by_identity = {
            (chunk.chunk_id, chunk.version_id, chunk.content_hash): chunk
            for chunk in request.chunks
        }
        candidate_ids: set[str] = set()
        covered_identities: set[tuple[str, str, str]] = set()
        previous_rank = 0

        for candidate in candidates:
            if candidate.evidence_id in candidate_ids:
                raise ValueError(
                    f"evidence transformer 返回了重复 evidence_id：{candidate.evidence_id}"
                )
            candidate_ids.add(candidate.evidence_id)

            primary_rank = candidate.primary_chunk.rank
            if primary_rank < previous_rank:
                raise ValueError("evidence transformer 不应改变候选的检索排序")
            previous_rank = primary_rank

            for source in candidate.sources:
                identity = (
                    source.chunk.chunk_id,
                    source.chunk.version_id,
                    source.chunk.content_hash,
                )
                if identity not in original_by_identity:
                    raise ValueError(
                        "evidence transformer 返回了候选集合外的来源："
                        f"{source.chunk.chunk_id}"
                    )
                covered_identities.add(identity)

        missing_identities = set(original_by_identity) - covered_identities
        if missing_identities:
            missing_chunk_ids = sorted(identity[0] for identity in missing_identities)
            raise ValueError(
                "evidence transformer 没有保留全部候选来源："
                + ", ".join(missing_chunk_ids)
            )
