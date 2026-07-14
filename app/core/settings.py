"""应用配置。

EnvSettings 只负责读取 .env 和环境变量中的敏感配置。
ProjectSettings 负责读取 settings.toml，适合结构化工程配置。
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.errors import AppError, ErrorCode


class EnvSettings(BaseSettings):
    """从环境变量读取的敏感配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        validate_default=True,
    )

    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
        description="OpenAI API 密钥",
    )

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def normalize_optional_secret(cls, value: object) -> object:
        """把空字符串视为未配置密钥。"""

        if isinstance(value, str) and not value.strip():
            return None
        return value

    @classmethod
    def from_env(cls) -> "EnvSettings":
        """从环境变量读取配置。

        业务入口使用这个方法，把 pydantic 的 ValidationError 转换成项目统一错误。
        """
        try:
            return cls()
        except ValidationError as exc:
            raise AppError(
                ErrorCode.INVALID_CONFIG, _format_validation_error(exc)
            ) from exc


class LoaderSettings(BaseModel):
    """本地文档 loader 的结构化配置。"""

    recursive: bool = True
    ignored_dir_names: frozenset[str] = Field(
        default_factory=lambda: frozenset(
            {".git", ".idea", "__pycache__", ".tmp_tests"}
        )
    )
    ignored_relative_paths: tuple[str, ...] = ("data/indexes",)
    skip_hidden_paths: bool = True
    temporary_file_prefixes: tuple[str, ...] = ("~$",)
    temporary_file_suffixes: tuple[str, ...] = (".tmp", ".part", ".crdownload")


class PdfCleanerSettings(BaseModel):
    """PDF 文本清洗器的结构化配置。"""

    edge_line_count: int = Field(default=2, gt=0)
    min_repeat_ratio: float = Field(default=0.6, gt=0, le=1)
    min_line_length: int = Field(default=3, gt=0)
    max_line_length: int = Field(default=120, gt=0)

    @model_validator(mode="after")
    def validate_line_length_window(self) -> "PdfCleanerSettings":
        """校验页眉页脚候选行长度窗口。"""

        if self.max_line_length < self.min_line_length:
            raise ValueError(
                f"max_line_length 必须大于等于 min_line_length，当前 max_line_length={self.max_line_length}，"
                f"min_line_length={self.min_line_length}"
            )
        return self


class IngestionReportSettings(BaseModel):
    """文档摄取报告的结构化配置。"""

    output_dir: Path = Field(
        default=Path("logs"), description="摄取报告 JSON 的输出目录"
    )


class ChunkingReportSettings(BaseModel):
    """chunking 质量报告配置。"""

    output_dir: Path = Field(
        default=Path("logs"), description="chunking 报告 JSON 的输出目录"
    )


class ChunkingSettings(BaseModel):
    """文本切分及其质量报告的结构化配置。"""

    strategy: str = Field(
        default="section_aware",
        min_length=1,
        description="chunking 策略",
    )
    chunk_size: int = Field(default=600, gt=0, description="每个 chunk 的目标长度")
    chunk_overlap: int = Field(default=100, ge=0, description="相邻 chunk 的重叠长度")
    tokenizer: Literal["char_approx", "simple_regex"] = Field(
        default="char_approx", description="token 估算方式"
    )
    report: ChunkingReportSettings = Field(default_factory=ChunkingReportSettings)

    @model_validator(mode="after")
    def validate_chunk_window(self) -> "ChunkingSettings":
        """校验 chunking 窗口。"""

        self.strategy = self.strategy.strip()
        if not self.strategy:
            raise ValueError("strategy 不能为空")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap 必须小于 chunk_size，当前 chunk_overlap={self.chunk_overlap}，"
                f"chunk_size={self.chunk_size}"
            )
        return self


class EmbeddingSettings(BaseModel):
    """Embedding 模型的结构化配置。

    provider、model、dimension 这类配置会影响索引版本，因此放在 TOML 中统一记录。
    API key 不放在这里，由 EnvSettings 从环境变量读取。
    """

    provider: Literal["mock", "openai"] = Field(
        default="mock", description="embedding 服务提供方"
    )
    model: str = Field(
        default="mock-hash-embedding", min_length=1, description="embedding 模型名称"
    )
    dimension: int = Field(default=16, gt=0, description="embedding 向量维度")
    batch_size: int = Field(default=32, gt=0, description="embedding 批处理大小")
    timeout_seconds: float = Field(
        default=30.0, gt=0, description="embedding 请求超时时间"
    )
    max_retries: int = Field(default=2, ge=0, description="embedding 最大重试次数")

    @model_validator(mode="after")
    def validate_text_fields(self) -> "EmbeddingSettings":
        """清理并校验字符串字段。"""

        self.model = self.model.strip()
        if not self.model:
            raise ValueError("model 不能为空")
        return self


class VectorRepositorySettings(BaseModel):
    """向量持久化的结构化配置。"""

    type: Literal["memory", "local_json"] = Field(
        default="memory", description="向量存储类型"
    )
    index_dir: Path = Field(default=Path("data/indexes"), description="索引根目录")
    collection_name: str = Field(
        default="papers_baseline", min_length=1, description="向量集合名称"
    )
    distance_metric: Literal["cosine"] = Field(
        default="cosine", description="向量相似度算法"
    )
    persist: bool = Field(default=False, description="是否持久化向量索引")

    @model_validator(mode="after")
    def validate_collection_name(self) -> "VectorRepositorySettings":
        """清理并校验 collection 名称。"""

        self.collection_name = self.collection_name.strip()
        if not self.collection_name:
            raise ValueError("collection_name 不能为空")
        return self


class IndexBuilderSettings(BaseModel):
    """索引构建流程配置。"""

    manifest_filename: str = Field(
        default="manifest.json", min_length=1, description="索引 manifest 文件名"
    )
    build_report_filename: str = Field(
        default="index_build_report.json",
        min_length=1,
        description="索引构建报告文件名",
    )
    skip_existing: bool = Field(
        default=True, description="是否跳过已经写入向量库的 chunk"
    )
    fail_on_empty_chunk: bool = Field(
        default=True, description="遇到空 chunk 时是否直接失败"
    )

    @model_validator(mode="after")
    def validate_filenames(self) -> "IndexBuilderSettings":
        """清理并校验文件名字段。"""

        self.manifest_filename = self.manifest_filename.strip()
        self.build_report_filename = self.build_report_filename.strip()
        if not self.manifest_filename:
            raise ValueError("manifest_filename 不能为空")
        if not self.build_report_filename:
            raise ValueError("build_report_filename 不能为空")
        return self


class TokenizerSettings(BaseModel):
    """检索分词器的结构化配置。"""

    strategy: str = Field(
        default="regex",
        min_length=1,
        description="BM25 索引与查询共同使用的分词策略",
    )

    @model_validator(mode="after")
    def validate_strategy(self) -> "TokenizerSettings":
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

    candidate_multiplier: int = Field(
        default=3,
        ge=1,
        description="每个召回源相对最终 top_k 的候选集扩张倍数",
    )
    rrf_rank_constant: int = Field(
        default=60,
        gt=0,
        description="RRF 排名常数，控制靠前排名的影响强度",
    )
    vector_weight: float = Field(default=1.0, gt=0, description="向量召回权重")
    bm25_weight: float = Field(default=1.0, gt=0, description="BM25 召回权重")


class RerankingSettings(BaseModel):
    """候选重排序阶段的结构化配置。"""

    enabled: bool = Field(default=False, description="是否在候选召回后执行 rerank")
    strategy: str = Field(default="lexical", min_length=1, description="reranker 策略名")
    candidate_limit: int = Field(default=12, gt=0, description="rerank 前候选召回上限")
    batch_size: int = Field(default=8, gt=0, description="reranker 批处理大小")
    failure_mode: Literal["fail_open", "fail_closed"] = Field(
        default="fail_open",
        description="reranker 失败时沿用原排序或终止请求",
    )

    @model_validator(mode="after")
    def validate_strategy(self) -> "RerankingSettings":
        """清理并校验 reranker 策略名。"""

        self.strategy = self.strategy.strip()
        if not self.strategy:
            raise ValueError("reranking strategy 不能为空")
        return self


class TokenEstimatorSettings(BaseModel):
    """模型上下文 token 估算器的结构化配置。"""

    strategy: str = Field(default="regex", min_length=1, description="token estimator 策略名")

    @model_validator(mode="after")
    def validate_strategy(self) -> "TokenEstimatorSettings":
        """清理并校验 token estimator 策略名。"""

        self.strategy = self.strategy.strip()
        if not self.strategy:
            raise ValueError("token estimator strategy 不能为空")
        return self


class EvidenceTransformationSettings(BaseModel):
    """候选证据变换阶段的结构化配置。"""

    enabled: bool = Field(
        default=True,
        description="是否在 ContextPacker 前执行候选证据变换",
    )
    strategy: str = Field(
        default="passthrough",
        min_length=1,
        description="evidence transformer 策略名",
    )
    failure_mode: Literal["fail_open", "fail_closed"] = Field(
        default="fail_open",
        description="证据变换失败时沿用原始候选或终止请求",
    )

    @model_validator(mode="after")
    def validate_strategy(self) -> "EvidenceTransformationSettings":
        """清理并校验 transformer 策略名。"""

        self.strategy = self.strategy.strip()
        if not self.strategy:
            raise ValueError("evidence transformation strategy 不能为空")
        return self


class ContextPackingSettings(BaseModel):
    """检索结果进入生成阶段前的 token-aware 上下文组织配置。"""

    model_context_window: int = Field(
        default=4096,
        gt=0,
        description="目标生成模型的上下文窗口大小",
    )
    max_context_tokens: int = Field(
        default=1800,
        gt=0,
        description="资料上下文最多使用的 token 数",
    )
    reserved_prompt_tokens: int = Field(
        default=200,
        gt=0,
        description="system prompt 与固定格式的预留 token 数",
    )
    reserved_output_tokens: int = Field(
        default=512,
        gt=0,
        description="为模型回答预留的 token 数",
    )
    safety_margin_tokens: int = Field(
        default=64,
        gt=0,
        description="避免估算误差导致超窗的安全余量",
    )
    max_chunks_per_document: int = Field(
        default=2,
        gt=0,
        description="单篇文档最多贡献多少个上下文候选",
    )
    token_estimator: TokenEstimatorSettings = Field(default_factory=TokenEstimatorSettings)
    evidence_transformation: EvidenceTransformationSettings = Field(
        default_factory=EvidenceTransformationSettings
    )

    @model_validator(mode="after")
    def validate_context_window(self) -> "ContextPackingSettings":
        """校验上下文 token 预算关系。"""

        if self.max_context_tokens > self.model_context_window:
            raise ValueError("max_context_tokens 不能大于 model_context_window")
        return self


class RetrievalReportSettings(BaseModel):
    """Retrieval 执行报告的结构化配置。"""

    enabled: bool = Field(default=False, description="是否写入 retrieval JSON 报告")
    output_dir: Path = Field(
        default=Path("logs/retrieval"),
        description="Retrieval 报告输出目录",
    )
    include_result_text: bool = Field(
        default=False,
        description="报告中是否包含检索文本预览",
    )
    result_preview_chars: int = Field(
        default=160,
        gt=0,
        description="检索文本预览最大字符数",
    )
    fail_on_write_error: bool = Field(
        default=False,
        description="报告写入失败时是否让检索请求失败",
    )


class RetrievalSettings(BaseModel):
    """检索子系统结构化配置。"""

    strategy: str = Field(default="vector", min_length=1, description="默认检索策略")
    top_k: int = Field(default=3, gt=0, description="默认返回的候选 chunk 数量")
    tokenizer: TokenizerSettings = Field(default_factory=TokenizerSettings)
    bm25: BM25Settings = Field(default_factory=BM25Settings)
    hybrid: HybridRetrievalSettings = Field(default_factory=HybridRetrievalSettings)
    reranking: RerankingSettings = Field(default_factory=RerankingSettings)
    context_packing: ContextPackingSettings = Field(
        default_factory=ContextPackingSettings
    )
    report: RetrievalReportSettings = Field(default_factory=RetrievalReportSettings)
    deduplicate_by_chunk_id: bool = Field(
        default=True, description="检索结果是否按 chunk_id 去重"
    )

    @model_validator(mode="after")
    def validate_strategy(self) -> "RetrievalSettings":
        """清理策略名称；具体合法性由 RetrieverRegistry 校验。"""

        self.strategy = self.strategy.strip()
        if not self.strategy:
            raise ValueError("strategy 不能为空")
        return self


class CleaningSettings(BaseModel):
    """文档清洗阶段的结构化配置。"""

    pdf: PdfCleanerSettings = Field(default_factory=PdfCleanerSettings)


class IngestionSettings(BaseModel):
    """文档加载、清洗、切分与摄取报告配置。"""

    loader: LoaderSettings = Field(default_factory=LoaderSettings)
    cleaning: CleaningSettings = Field(default_factory=CleaningSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    report: IngestionReportSettings = Field(default_factory=IngestionReportSettings)


class IndexingSettings(BaseModel):
    """Embedding、持久化与索引构建配置。"""

    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_repository: VectorRepositorySettings = Field(
        default_factory=VectorRepositorySettings
    )
    builder: IndexBuilderSettings = Field(default_factory=IndexBuilderSettings)


class ProjectSettings(BaseModel):
    """从 settings.toml 读取的结构化工程配置。"""

    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    indexing: IndexingSettings = Field(default_factory=IndexingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)

    @classmethod
    def from_toml(cls, path: Path | str = Path("settings.toml")) -> "ProjectSettings":
        """从 TOML 文件读取结构化配置。"""

        config_path = Path(path)
        if not config_path.exists():
            return cls()

        try:
            data = tomllib.loads(config_path.read_text(encoding="utf-8"))
            return cls.model_validate(data)
        except (OSError, tomllib.TOMLDecodeError, ValidationError) as exc:
            if isinstance(exc, ValidationError):
                message = _format_validation_error(exc)
            else:
                message = f"配置文件读取失败：{exc}"
            raise AppError(ErrorCode.INVALID_CONFIG, message) from exc


def _format_validation_error(error: ValidationError) -> str:
    """把 pydantic 的校验错误整理成更适合 CLI/API 展示的中文消息。"""

    messages: list[str] = []
    for item in error.errors():
        location = ".".join(str(part) for part in item.get("loc", ())) or "settings"
        messages.append(f"{location}: {item.get('msg')}")
    return "配置校验失败：" + "；".join(messages)
