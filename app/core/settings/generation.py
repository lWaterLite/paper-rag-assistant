"""查询规划与回答生成领域的外部 Settings。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class LlmSettings(BaseModel):
    """LLM provider 的非敏感外部配置。"""

    provider: str = Field(
        default="mock", min_length=1, description="LLM provider 注册名"
    )
    model: str = Field(
        default="mock-grounded-json", min_length=1, description="模型标识"
    )
    base_url: str | None = Field(
        default=None, description="完整 Chat Completions endpoint"
    )
    timeout_seconds: float = Field(default=30.0, gt=0, description="单次模型调用超时")
    max_retries: int = Field(default=1, ge=0, description="可重试模型请求次数")

    @model_validator(mode="after")
    def normalize_names(self) -> LlmSettings:
        """清理 provider、模型名和可选 endpoint。"""

        self.provider = self.provider.strip()
        self.model = self.model.strip()
        self.base_url = self.base_url.strip() if self.base_url else None
        if not self.provider:
            raise ValueError("provider 不能为空")
        if not self.model:
            raise ValueError("model 不能为空")
        return self


class QueryPlanningSettings(BaseModel):
    """查询改写与多查询检索的外部配置。"""

    enabled: bool = Field(default=True, description="是否在检索前执行查询规划")
    strategy: str = Field(
        default="rule_based", min_length=1, description="QueryPlanner 注册名"
    )
    multi_query_enabled: bool = Field(
        default=False, description="是否执行多个互补 query"
    )
    max_additional_queries: int = Field(
        default=2, ge=0, le=8, description="额外 query 上限"
    )
    hyde_enabled: bool = Field(
        default=False, description="是否允许生成 HyDE 检索辅助文本"
    )
    failure_mode: Literal["fail_open", "fail_closed"] = Field(
        default="fail_open",
        description="改写失败时退回原始问题或终止请求",
    )

    @model_validator(mode="after")
    def normalize_strategy(self) -> QueryPlanningSettings:
        """清理 QueryPlanner 策略名称。"""

        self.strategy = self.strategy.strip()
        if not self.strategy:
            raise ValueError("strategy 不能为空")
        return self


class AnswerGenerationSettings(BaseModel):
    """回答生成行为的外部配置。"""

    temperature: float = Field(default=0.0, ge=0, le=2, description="模型采样温度")
    prompt_version: str = Field(
        default="rag_answer_v2", min_length=1, description="Prompt 版本"
    )
    default_language: str = Field(
        default="中文", min_length=1, description="默认回答语言"
    )
    invalid_output_mode: Literal["fail_closed", "abstain"] = Field(
        default="fail_closed",
        description="模型输出或 citation 校验失败时的处理方式",
    )

    @model_validator(mode="after")
    def normalize_text(self) -> AnswerGenerationSettings:
        """清理可展示文本配置。"""

        self.prompt_version = self.prompt_version.strip()
        self.default_language = self.default_language.strip()
        if not self.prompt_version or not self.default_language:
            raise ValueError("prompt_version 和 default_language 不能为空")
        return self


class CitationValidationSettings(BaseModel):
    """回答级引用校验的外部配置。"""

    require_citations_when_evidence: bool = Field(
        default=True,
        description="有证据的非拒答回答必须包含引用",
    )
    require_inline_ids_match_payload: bool = Field(
        default=True,
        description="正文 citation id 必须与结构化字段一致",
    )
    require_abstention_reason: bool = Field(
        default=True,
        description="拒答必须解释资料不足原因",
    )


class GenerationSettings(BaseModel):
    """生成子系统的结构化外部配置。"""

    llm: LlmSettings = Field(default_factory=LlmSettings)
    query_planning: QueryPlanningSettings = Field(default_factory=QueryPlanningSettings)
    answering: AnswerGenerationSettings = Field(
        default_factory=AnswerGenerationSettings
    )
    citation_validation: CitationValidationSettings = Field(
        default_factory=CitationValidationSettings
    )
