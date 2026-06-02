"""应用配置。

这里使用 pydantic-settings 管理项目配置。
它负责从环境变量和 .env 文件读取配置，并用 pydantic 完成类型转换与字段校验。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.errors import AppError, ErrorCode


class Settings(BaseSettings):
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
    def validate_chunk_window(self) -> "Settings":
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
    def from_env(cls) -> "Settings":
        """从环境变量读取配置。

        业务入口使用这个方法，可以把 pydantic 的 ValidationError 转换成项目统一错误。
        测试或内部代码也可以直接使用 Settings(...)，直接获得 pydantic 的标准校验行为。
        """
        try:
            return cls()
        except ValidationError as exc:
            raise AppError(ErrorCode.INVALID_CONFIG, _format_validation_error(exc)) from exc


def _format_validation_error(error: ValidationError) -> str:
    """把 pydantic 的校验错误整理成更适合 CLI/API 展示的中文消息。"""

    messages: list[str] = []
    for item in error.errors():
        location = ".".join(str(part) for part in item.get("loc", ())) or "settings"
        messages.append(f"{location}: {item.get('msg')}")
    return "配置校验失败：" + "；".join(messages)
