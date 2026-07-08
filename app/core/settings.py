"""应用配置。

EnvSettings 负责读取 .env 和环境变量，适合环境相关、敏感或部署覆盖项。
ProjectSettings 负责读取 settings.toml，适合结构化工程配置。
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.errors import AppError, ErrorCode


class EnvSettings(BaseSettings):
    """RAG pipeline 的基础配置。

    这里故意只保留子模块 1 需要理解的配置项，避免一开始就陷入外部模型和数据库细节。
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_default=True,
    )

    chunk_size: int = Field(default=500, gt=0, description="每个 chunk 的目标字符长度")
    chunk_overlap: int = Field(default=80, ge=0, description="相邻 chunk 的重叠字符数")
    top_k: int = Field(default=3, gt=0, description="检索阶段返回的候选 chunk 数量")
    max_context_chars: int = Field(default=1800, gt=0, description="进入生成阶段的最大上下文字符数")
    mock_embedding_dimension: int = Field(default=16, gt=0, description="mock embedding 的向量维度")
    require_citation: bool = Field(default=True, description="回答是否要求包含引用")
    retrieval_strategy: Literal["vector", "bm25", "hybrid"] = Field(default="vector", description="检索策略")
    index_storage_path: Path = Field(default=Path("data/indexes"), description="索引持久化目录")
    debug_trace: bool = Field(default=False, description="是否在响应中返回完整 trace")

    @model_validator(mode="after")
    def validate_chunk_window(self) -> "EnvSettings":
        """校验多个配置项之间的关系。

        Field 适合校验单个字段，例如大于 0。
        model_validator 适合校验跨字段约束，例如 overlap 必须小于 chunk_size。
        """

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap 必须小于 chunk_size，当前 chunk_overlap={self.chunk_overlap}，"
                f"chunk_size={self.chunk_size}"
            )
        return self

    @classmethod
    def from_env(cls) -> "EnvSettings":
        """从环境变量读取配置。

        业务入口使用这个方法，可以把 pydantic 的 ValidationError 转换成项目统一错误。
        测试或内部代码也可以直接使用 EnvSettings(...)，直接获得 pydantic 的标准校验行为。
        """
        try:
            return cls()
        except ValidationError as exc:
            raise AppError(ErrorCode.INVALID_CONFIG, _format_validation_error(exc)) from exc


class LoaderSettings(BaseModel):
    """本地文档 loader 的结构化配置。"""

    recursive: bool = True
    ignored_dir_names: frozenset[str] = Field(
        default_factory=lambda: frozenset({".git", ".idea", "__pycache__", ".tmp_tests"})
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

    output_dir: Path = Field(default=Path("logs"), description="摄取报告 JSON 的输出目录")


class ChunkingSettings(BaseModel):
    """文本切分的结构化配置。"""

    strategy: str = Field(
        default="section_aware",
        min_length=1,
        description="chunking 策略",
    )
    chunk_size: int = Field(default=600, gt=0, description="每个 chunk 的目标长度")
    chunk_overlap: int = Field(default=100, ge=0, description="相邻 chunk 的重叠长度")
    tokenizer: Literal["char_approx", "simple_regex"] = Field(default="char_approx", description="token 估算方式")

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


class ChunkingReportSettings(BaseModel):
    """chunking 质量报告配置。"""

    output_dir: Path = Field(default=Path("logs"), description="chunking 报告 JSON 的输出目录")


class EmbeddingSettings(BaseModel):
    """Embedding 模型的结构化配置。

    provider、model、dimension 这类配置会影响索引版本，因此放在 TOML 中统一记录。
    API key 不放在这里，只记录应读取哪个环境变量。
    """

    provider: Literal["mock", "openai"] = Field(default="mock", description="embedding 服务提供方")
    model: str = Field(default="mock-hash-embedding", min_length=1, description="embedding 模型名称")
    dimension: int = Field(default=16, gt=0, description="embedding 向量维度")
    batch_size: int = Field(default=32, gt=0, description="embedding 批处理大小")
    timeout_seconds: float = Field(default=30.0, gt=0, description="embedding 请求超时时间")
    max_retries: int = Field(default=2, ge=0, description="embedding 最大重试次数")
    api_key_env_name: str = Field(default="OPENAI_API_KEY", min_length=1, description="真实 provider 的 API key 环境变量名")

    @model_validator(mode="after")
    def validate_text_fields(self) -> "EmbeddingSettings":
        """清理并校验字符串字段。"""

        self.model = self.model.strip()
        self.api_key_env_name = self.api_key_env_name.strip()
        if not self.model:
            raise ValueError("model 不能为空")
        if not self.api_key_env_name:
            raise ValueError("api_key_env_name 不能为空")
        return self


class VectorRepositorySettings(BaseModel):
    """向量持久化的结构化配置。"""

    type: Literal["memory", "local_json"] = Field(default="memory", description="向量存储类型")
    index_dir: Path = Field(default=Path("data/indexes"), description="索引根目录")
    collection_name: str = Field(default="papers_baseline", min_length=1, description="向量集合名称")
    distance_metric: Literal["cosine"] = Field(default="cosine", description="向量相似度算法")
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

    manifest_filename: str = Field(default="manifest.json", min_length=1, description="索引 manifest 文件名")
    build_report_filename: str = Field(default="index_build_report.json", min_length=1, description="索引构建报告文件名")
    skip_existing: bool = Field(default=True, description="是否跳过已经写入向量库的 chunk")
    fail_on_empty_chunk: bool = Field(default=True, description="遇到空 chunk 时是否直接失败")

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


class RetrievalSettings(BaseModel):
    """检索子系统结构化配置。"""

    bm25_k1: float = Field(default=1.5, gt=0, description="BM25 词频饱和参数")
    bm25_b: float = Field(default=0.75, ge=0, le=1, description="BM25 文档长度归一化参数")
    deduplicate_by_chunk_id: bool = Field(default=True, description="检索结果是否按 chunk_id 去重")


class ProjectSettings(BaseModel):
    """从 settings.toml 读取的结构化工程配置。"""

    loader: LoaderSettings = Field(default_factory=LoaderSettings)
    pdf_cleaner: PdfCleanerSettings = Field(default_factory=PdfCleanerSettings)
    ingestion_report: IngestionReportSettings = Field(default_factory=IngestionReportSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    chunking_report: ChunkingReportSettings = Field(default_factory=ChunkingReportSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_repository: VectorRepositorySettings = Field(default_factory=VectorRepositorySettings)
    index_builder: IndexBuilderSettings = Field(default_factory=IndexBuilderSettings)
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
