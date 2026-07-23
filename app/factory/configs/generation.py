"""Generation Settings 到运行期 Config 的适配。"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.settings import GenerationSettings, RetrievalSettings
from app.generation.configuration import CitationValidationConfig, GenerationConfig
from app.llm import LlmClientConfig
from app.retrieval.query import QueryPlanningConfig


@dataclass(frozen=True, slots=True)
class GenerationConfigAdapter:
    """将生成 Settings 与上下文窗口 Settings 组合成不可变 Config 快照。"""

    generation_settings: GenerationSettings
    retrieval_settings: RetrievalSettings
    llm: LlmClientConfig = field(init=False)
    query_planning: QueryPlanningConfig = field(init=False)
    answering: GenerationConfig = field(init=False)
    citation_validation: CitationValidationConfig = field(init=False)

    def __post_init__(self) -> None:
        generation = self.generation_settings
        llm_settings = generation.llm
        object.__setattr__(
            self,
            "llm",
            LlmClientConfig(
                provider=llm_settings.provider,
                model=llm_settings.model,
                base_url=llm_settings.base_url,
                timeout_seconds=llm_settings.timeout_seconds,
                max_retries=llm_settings.max_retries,
            ),
        )
        planning = generation.query_planning
        object.__setattr__(
            self,
            "query_planning",
            QueryPlanningConfig(
                enabled=planning.enabled,
                strategy=planning.strategy,
                multi_query_enabled=planning.multi_query_enabled,
                max_additional_queries=planning.max_additional_queries,
                hyde_enabled=planning.hyde_enabled,
                failure_mode=planning.failure_mode,
            ),
        )
        answering = generation.answering
        object.__setattr__(
            self,
            "answering",
            GenerationConfig(
                model=llm_settings.model,
                temperature=answering.temperature,
                max_output_tokens=self.retrieval_settings.context_packing.reserved_output_tokens,
                timeout_seconds=llm_settings.timeout_seconds,
                prompt_version=answering.prompt_version,
                default_language=answering.default_language,
                invalid_output_mode=answering.invalid_output_mode,
            ),
        )
        validation = generation.citation_validation
        object.__setattr__(
            self,
            "citation_validation",
            CitationValidationConfig(
                require_citations_when_evidence=validation.require_citations_when_evidence,
                require_inline_ids_match_payload=validation.require_inline_ids_match_payload,
                require_abstention_reason=validation.require_abstention_reason,
            ),
        )
