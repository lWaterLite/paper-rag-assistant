"""Retrieval Settings 到运行时 Config 的适配。"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.settings import RetrievalSettings
from app.retrieval.configuration import BM25Config, HybridRetrievalConfig, RetrievalConfig
from app.retrieval.configuration.postprocessing import PostProcessingConfig
from app.retrieval.context import ContextPackerConfig
from app.retrieval.context.evidence_transformers import EvidenceTransformationConfig
from app.retrieval.context.token_estimators import TokenEstimatorConfig
from app.retrieval.rerankers import RerankingConfig
from app.retrieval.reporting import RetrievalReportConfig
from app.retrieval.tokenizers import TokenizerConfig


@dataclass(frozen=True, slots=True)
class RetrievalConfigAdapter:
    """将 RetrievalSettings 一次转换为可复用的运行时 Config 快照。"""

    settings: RetrievalSettings
    retrieval: RetrievalConfig = field(init=False)
    tokenizer: TokenizerConfig = field(init=False)
    hybrid: HybridRetrievalConfig = field(init=False)
    reranking: RerankingConfig = field(init=False)
    token_estimator: TokenEstimatorConfig = field(init=False)
    context_packer: ContextPackerConfig = field(init=False)
    evidence_transformation: EvidenceTransformationConfig = field(init=False)
    postprocessing: PostProcessingConfig = field(init=False)
    report: RetrievalReportConfig = field(init=False)

    def __post_init__(self) -> None:
        settings = self.settings
        retrieval_config = RetrievalConfig(
            strategy=settings.strategy,
            top_k=settings.top_k,
            bm25=BM25Config(k1=settings.bm25.k1, b=settings.bm25.b),
            deduplicate_by_chunk_id=settings.deduplicate_by_chunk_id,
        )
        object.__setattr__(self, "retrieval", retrieval_config)
        object.__setattr__(
            self,
            "tokenizer",
            TokenizerConfig(strategy=settings.tokenizer.strategy),
        )
        object.__setattr__(
            self,
            "hybrid",
            HybridRetrievalConfig(
                candidate_multiplier=settings.hybrid.candidate_multiplier,
                rrf_rank_constant=settings.hybrid.rrf_rank_constant,
                vector_weight=settings.hybrid.vector_weight,
                bm25_weight=settings.hybrid.bm25_weight,
            ),
        )
        object.__setattr__(
            self,
            "reranking",
            RerankingConfig(
                enabled=settings.reranking.enabled,
                strategy=settings.reranking.strategy,
                candidate_limit=settings.reranking.candidate_limit,
                batch_size=settings.reranking.batch_size,
                failure_mode=settings.reranking.failure_mode,
            ),
        )
        object.__setattr__(
            self,
            "token_estimator",
            TokenEstimatorConfig(
                strategy=settings.context_packing.token_estimator.strategy
            ),
        )
        object.__setattr__(
            self,
            "context_packer",
            ContextPackerConfig(
                model_context_window=settings.context_packing.model_context_window,
                max_context_tokens=settings.context_packing.max_context_tokens,
                reserved_prompt_tokens=settings.context_packing.reserved_prompt_tokens,
                reserved_output_tokens=settings.context_packing.reserved_output_tokens,
                safety_margin_tokens=settings.context_packing.safety_margin_tokens,
                max_chunks_per_document=settings.context_packing.max_chunks_per_document,
            ),
        )
        object.__setattr__(
            self,
            "evidence_transformation",
            EvidenceTransformationConfig(
                enabled=settings.context_packing.evidence_transformation.enabled,
                strategy=settings.context_packing.evidence_transformation.strategy,
                failure_mode=settings.context_packing.evidence_transformation.failure_mode,
            ),
        )
        object.__setattr__(
            self,
            "postprocessing",
            PostProcessingConfig(
                retrieval=retrieval_config,
                reranking=self.reranking,
                context_packing=self.context_packer,
                evidence_transformation=self.evidence_transformation,
            ),
        )
        report_settings = settings.report
        object.__setattr__(
            self,
            "report",
            RetrievalReportConfig(
                enabled=report_settings.enabled,
                output_dir=report_settings.output_dir,
                include_result_text=report_settings.include_result_text,
                result_preview_chars=report_settings.result_preview_chars,
                fail_on_write_error=report_settings.fail_on_write_error,
            ),
        )
