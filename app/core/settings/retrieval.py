"""在线检索领域的外部 Settings。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TokenizerSettings(BaseModel):
    """检索分词器的结构化配置。"""

    strategy: str = Field(
        default="regex",
        min_length=1,
        description="BM25 索引与查询共同使用的分词策略",
    )

    @model_validator(mode="after")
    def validate_strategy(self) -> TokenizerSettings:
        """清理并校验分词器策略名称。"""

        self.strategy = self.strategy.strip()
        if not self.strategy:
            raise ValueError("strategy 不能为空")
        return self


class BM25Settings(BaseModel):
    """BM25 检索算法的结构化配置。"""

    k1: float = Field(default=1.5, gt=0, description="词频饱和参数")
    b: float = Field(default=0.75, ge=0, le=1, description="文档长度归一化参数")


class HybridRetrievalSettings(BaseModel):
    """Hybrid Retriever 与 RRF 融合的结构化配置。"""

    candidate_multiplier: int = Field(default=3, ge=1, description="候选集扩张倍数")
    rrf_rank_constant: int = Field(default=60, gt=0, description="RRF 排名常数")
    vector_weight: float = Field(default=1.0, gt=0, description="向量召回权重")
    bm25_weight: float = Field(default=1.0, gt=0, description="BM25 召回权重")


class RerankingSettings(BaseModel):
    """候选重排序阶段的结构化配置。"""

    enabled: bool = Field(default=False, description="是否在候选召回后执行 rerank")
    strategy: str = Field(
        default="lexical", min_length=1, description="reranker 策略名"
    )
    candidate_limit: int = Field(default=12, gt=0, description="rerank 前候选上限")
    batch_size: int = Field(default=8, gt=0, description="reranker 批处理大小")
    failure_mode: Literal["fail_open", "fail_closed"] = Field(
        default="fail_open",
        description="reranker 失败时沿用原排序或终止请求",
    )

    @model_validator(mode="after")
    def validate_strategy(self) -> RerankingSettings:
        """清理并校验 reranker 策略名。"""

        self.strategy = self.strategy.strip()
        if not self.strategy:
            raise ValueError("reranking strategy 不能为空")
        return self


class TokenEstimatorSettings(BaseModel):
    """模型上下文 token 估算器的结构化配置。"""

    strategy: str = Field(
        default="regex", min_length=1, description="token estimator 策略名"
    )

    @model_validator(mode="after")
    def validate_strategy(self) -> TokenEstimatorSettings:
        """清理并校验 token estimator 策略名。"""

        self.strategy = self.strategy.strip()
        if not self.strategy:
            raise ValueError("token estimator strategy 不能为空")
        return self


class EvidenceTransformationSettings(BaseModel):
    """候选证据变换阶段的结构化配置。"""

    enabled: bool = Field(default=True, description="是否执行候选证据变换")
    strategy: str = Field(default="passthrough", min_length=1, description="策略名")
    failure_mode: Literal["fail_open", "fail_closed"] = Field(
        default="fail_open",
        description="证据变换失败时沿用原始候选或终止请求",
    )

    @model_validator(mode="after")
    def validate_strategy(self) -> EvidenceTransformationSettings:
        """清理并校验 transformer 策略名。"""

        self.strategy = self.strategy.strip()
        if not self.strategy:
            raise ValueError("evidence transformation strategy 不能为空")
        return self


class ContextPackingSettings(BaseModel):
    """检索结果进入生成阶段前的 token-aware 上下文组织配置。"""

    model_context_window: int = Field(
        default=4096, gt=0, description="目标模型上下文窗口"
    )
    max_context_tokens: int = Field(default=1800, gt=0, description="资料上下文上限")
    reserved_prompt_tokens: int = Field(default=200, gt=0, description="Prompt 预留")
    reserved_output_tokens: int = Field(default=512, gt=0, description="输出预留")
    safety_margin_tokens: int = Field(default=64, gt=0, description="安全余量")
    max_chunks_per_document: int = Field(default=2, gt=0, description="单文档候选上限")
    token_estimator: TokenEstimatorSettings = Field(
        default_factory=TokenEstimatorSettings
    )
    evidence_transformation: EvidenceTransformationSettings = Field(
        default_factory=EvidenceTransformationSettings
    )

    @model_validator(mode="after")
    def validate_context_window(self) -> ContextPackingSettings:
        """校验基础上下文 token 预算关系。"""

        if self.max_context_tokens > self.model_context_window:
            raise ValueError("max_context_tokens 不能大于 model_context_window")
        return self


class RetrievalReportSettings(BaseModel):
    """Retrieval 执行报告的结构化配置。"""

    enabled: bool = Field(default=False, description="是否写入 retrieval JSON 报告")
    output_dir: Path = Field(default=Path("logs/retrieval"), description="报告输出目录")
    include_result_text: bool = Field(default=False, description="是否包含检索文本预览")
    result_preview_chars: int = Field(default=160, gt=0, description="预览字符数")
    fail_on_write_error: bool = Field(default=False, description="报告失败是否终止请求")


class RetrievalSettings(BaseModel):
    """检索子系统结构化配置。"""

    strategy: str = Field(default="vector", min_length=1, description="默认检索策略")
    top_k: int = Field(default=3, gt=0, description="默认返回候选数量")
    tokenizer: TokenizerSettings = Field(default_factory=TokenizerSettings)
    bm25: BM25Settings = Field(default_factory=BM25Settings)
    hybrid: HybridRetrievalSettings = Field(default_factory=HybridRetrievalSettings)
    reranking: RerankingSettings = Field(default_factory=RerankingSettings)
    context_packing: ContextPackingSettings = Field(
        default_factory=ContextPackingSettings
    )
    report: RetrievalReportSettings = Field(default_factory=RetrievalReportSettings)
    deduplicate_by_chunk_id: bool = Field(
        default=True, description="是否按 chunk_id 去重"
    )

    @model_validator(mode="after")
    def validate_strategy(self) -> RetrievalSettings:
        """清理策略名称；具体合法性由 RetrieverRegistry 校验。"""

        self.strategy = self.strategy.strip()
        if not self.strategy:
            raise ValueError("strategy 不能为空")
        return self
