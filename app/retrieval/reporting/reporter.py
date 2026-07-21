"""Retrieval 报告写入协调器。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.retrieval.reporting.config import RetrievalReportConfig
from app.retrieval.reporting.models import (
    RetrievalConfigSnapshot,
    RetrievalExecutionReport,
    RetrievalIndexSnapshot,
    RetrievalRuntimeSnapshot,
)
from app.retrieval.reporting.writer import RetrievalReportWriter
from app.retrieval.configuration.postprocessing import PostProcessingConfig
from app.retrieval.configuration.postprocessing.profile import PostProcessingProfile
from app.retrieval.configuration import RetrievalConfig
from app.retrieval.context import ContextPackerConfig
from app.retrieval.context.evidence_transformers import EvidenceTransformationConfig
from app.retrieval.rerankers import RerankingConfig


@dataclass(frozen=True, slots=True)
class RetrievalReportWriteResult:
    """报告写入结果，不让非关键报告失败静默消失。"""

    path: Path | None = None
    error_message: str | None = None
    fatal: bool = False


class RetrievalReporter:
    """持有报告策略、运行时快照和 writer。"""

    def __init__(
        self,
        *,
        config: RetrievalReportConfig,
        runtime_snapshot: RetrievalRuntimeSnapshot,
        writer: RetrievalReportWriter,
    ) -> None:
        self._config = config
        self._runtime_snapshot = runtime_snapshot
        self._writer = writer

    @classmethod
    def disabled(cls) -> "RetrievalReporter":
        """创建显式禁用的 reporter，适合独立组件测试。"""

        return cls(
            config=RetrievalReportConfig(enabled=False),
            runtime_snapshot=RetrievalRuntimeSnapshot(
                index=RetrievalIndexSnapshot(
                    index_id="unavailable",
                    schema_version=0,
                    status="unavailable",
                    artifact_definition_hash="",
                    document_set_hash="",
                    document_count=0,
                    chunk_count=0,
                    vector_count=0,
                    embedding_provider="unavailable",
                    embedding_model="unavailable",
                    embedding_dimension=0,
                    vector_repository_type="unavailable",
                    vector_collection_name="unavailable",
                    distance_metric="unavailable",
                ),
                config=RetrievalConfigSnapshot(
                    default_strategy="unavailable",
                    default_top_k=0,
                    deduplicate_by_chunk_id=False,
                    tokenizer_strategy="unavailable",
                    bm25_k1=0,
                    bm25_b=0,
                    hybrid_candidate_multiplier=0,
                    hybrid_rrf_rank_constant=0,
                    hybrid_vector_weight=0,
                    hybrid_bm25_weight=0,
                    postprocessing=PostProcessingProfile.from_config(
                        PostProcessingConfig(
                            retrieval=RetrievalConfig(top_k=1),
                            reranking=RerankingConfig(enabled=False),
                            context_packing=ContextPackerConfig(),
                            evidence_transformation=EvidenceTransformationConfig(),
                        )
                    ),
                    registered_strategies=(),
                ),
            ),
            writer=RetrievalReportWriter(),
        )

    @property
    def runtime_snapshot(self) -> RetrievalRuntimeSnapshot:
        """返回由 factory 固化的运行时快照。"""

        return self._runtime_snapshot

    @property
    def enabled(self) -> bool:
        """报告功能是否启用。"""

        return self._config.enabled

    def prepare_output_directory(self) -> None:
        """在流程启动阶段准备报告目录。"""

        if self._config.enabled:
            self._config.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, report: RetrievalExecutionReport) -> RetrievalReportWriteResult:
        """按配置写入报告，并返回明确的成功或失败结果。"""

        if not self._config.enabled:
            return RetrievalReportWriteResult()

        output_path = self._config.output_path(report.trace.trace_id)
        try:
            path = self._writer.write(report, output_path, self._config)
            return RetrievalReportWriteResult(path=path)
        except OSError as exc:
            return RetrievalReportWriteResult(
                error_message=str(exc),
                fatal=self._config.fail_on_write_error,
            )
