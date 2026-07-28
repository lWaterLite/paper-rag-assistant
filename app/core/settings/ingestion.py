"""文档摄取领域的外部 Settings。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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


class DocumentSourceAccessSettings(BaseModel):
    """文档导入入口允许访问的本地目录。"""

    allowed_source_dirs: tuple[Path, ...] = Field(
        default=(Path("data/raw"),),
        min_length=1,
        description="API 或其他受限入口允许导入的文档根目录",
    )

    @field_validator("allowed_source_dirs", mode="before")
    @classmethod
    def validate_allowed_source_dirs(cls, value: object) -> object:
        """拒绝空白目录项，避免它被 Path 转换为当前工作目录。"""

        if not isinstance(value, (list, tuple)):
            return value
        if not value:
            raise ValueError("allowed_source_dirs 至少需要一个目录")
        if any(not str(item).strip() for item in value):
            raise ValueError("allowed_source_dirs 不能包含空白目录")
        return value


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
        default=Path("logs"),
        description="摄取报告 JSON 的输出目录",
    )


class ChunkingReportSettings(BaseModel):
    """Chunking 质量报告配置。"""

    output_dir: Path = Field(
        default=Path("logs"),
        description="chunking 报告 JSON 的输出目录",
    )


class ChunkingSettings(BaseModel):
    """文本切分及其质量报告的结构化配置。"""

    strategy: str = Field(
        default="section_aware", min_length=1, description="chunking 策略"
    )
    chunk_size: int = Field(default=600, gt=0, description="每个 chunk 的目标长度")
    chunk_overlap: int = Field(default=100, ge=0, description="相邻 chunk 的重叠长度")
    tokenizer: Literal["char_approx", "simple_regex"] = Field(
        default="char_approx",
        description="token 估算方式",
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


class CleaningSettings(BaseModel):
    """文档清洗阶段的结构化配置。"""

    pdf: PdfCleanerSettings = Field(default_factory=PdfCleanerSettings)


class IngestionSettings(BaseModel):
    """文档加载、清洗、切分与摄取报告配置。"""

    loader: LoaderSettings = Field(default_factory=LoaderSettings)
    access: DocumentSourceAccessSettings = Field(
        default_factory=DocumentSourceAccessSettings
    )
    cleaning: CleaningSettings = Field(default_factory=CleaningSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    report: IngestionReportSettings = Field(default_factory=IngestionReportSettings)
