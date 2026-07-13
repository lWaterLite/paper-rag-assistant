"""检索后处理流程的稳定运行时摘要。"""

from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.postprocessing.config import PostProcessingConfig


@dataclass(frozen=True, slots=True)
class PostProcessingProfile:
    """记录一次应用组装所采用的后处理方案。

    Profile 面向报告和诊断，不负责创建业务对象，也不读取外部配置文件。
    """

    reranking_enabled: bool
    reranking_strategy: str | None
    reranking_failure_mode: str | None
    configured_candidate_limit: int | None
    default_candidate_limit: int
    candidate_limit_source: str
    default_top_k: int
    deduplicate_by_chunk_id: bool
    max_chunks_per_document: int
    model_context_window: int
    max_context_tokens: int
    reserved_prompt_tokens: int
    reserved_output_tokens: int
    safety_margin_tokens: int
    static_available_context_tokens: int
    effective_context_token_limit: int

    @classmethod
    def from_config(cls, config: PostProcessingConfig) -> "PostProcessingProfile":
        """从已通过组合校验的 Config 构建不可变摘要。"""

        context_packing = config.context_packing
        static_available_context_tokens = (
            context_packing.model_context_window
            - context_packing.reserved_prompt_tokens
            - context_packing.reserved_output_tokens
            - context_packing.safety_margin_tokens
        )
        reranking_enabled = config.reranking.enabled
        default_candidate_limit = (
            max(config.retrieval.top_k, config.reranking.candidate_limit)
            if reranking_enabled
            else config.retrieval.top_k
        )
        return cls(
            reranking_enabled=reranking_enabled,
            reranking_strategy=config.reranking.strategy if reranking_enabled else None,
            reranking_failure_mode=(
                config.reranking.failure_mode if reranking_enabled else None
            ),
            configured_candidate_limit=(
                config.reranking.candidate_limit if reranking_enabled else None
            ),
            default_candidate_limit=default_candidate_limit,
            candidate_limit_source=(
                "reranking_candidate_window" if reranking_enabled else "resolved_top_k"
            ),
            default_top_k=config.retrieval.top_k,
            deduplicate_by_chunk_id=config.retrieval.deduplicate_by_chunk_id,
            max_chunks_per_document=context_packing.max_chunks_per_document,
            model_context_window=context_packing.model_context_window,
            max_context_tokens=context_packing.max_context_tokens,
            reserved_prompt_tokens=context_packing.reserved_prompt_tokens,
            reserved_output_tokens=context_packing.reserved_output_tokens,
            safety_margin_tokens=context_packing.safety_margin_tokens,
            static_available_context_tokens=static_available_context_tokens,
            effective_context_token_limit=min(
                context_packing.max_context_tokens,
                static_available_context_tokens,
            ),
        )
