"""索引构建领域的外部 Settings。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class EmbeddingSettings(BaseModel):
    """Embedding 模型的结构化配置。"""

    provider: Literal["mock"] = Field(
        default="mock",
        description="当前内置 embedding 服务提供方",
    )
    model: str = Field(
        default="mock-hash-embedding",
        min_length=1,
        description="embedding 模型名称",
    )
    dimension: int = Field(default=16, gt=0, description="embedding 向量维度")
    batch_size: int = Field(default=32, gt=0, description="embedding 批处理大小")
    timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="embedding 请求超时时间",
    )
    max_retries: int = Field(default=2, ge=0, description="embedding 最大重试次数")

    @model_validator(mode="after")
    def validate_text_fields(self) -> EmbeddingSettings:
        """清理并校验字符串字段。"""

        self.model = self.model.strip()
        if not self.model:
            raise ValueError("model 不能为空")
        return self


class VectorRepositorySettings(BaseModel):
    """向量持久化的结构化配置。"""

    type: Literal["local_json"] = Field(
        default="local_json",
        description="向量持久化 Repository 类型",
    )
    index_dir: Path = Field(default=Path("data/indexes"), description="索引根目录")
    collection_name: str = Field(
        default="papers_baseline",
        min_length=1,
        description="向量集合名称",
    )
    distance_metric: Literal["cosine"] = Field(
        default="cosine",
        description="向量相似度算法",
    )

    @model_validator(mode="after")
    def validate_collection_name(self) -> VectorRepositorySettings:
        """清理并校验 collection 名称。"""

        self.collection_name = self.collection_name.strip()
        if not self.collection_name:
            raise ValueError("collection_name 不能为空")
        return self


class IndexBuilderSettings(BaseModel):
    """索引构建流程配置。"""

    manifest_filename: str = Field(
        default="manifest.json",
        min_length=1,
        description="索引 manifest 文件名",
    )
    build_report_filename: str = Field(
        default="index_build_report.json",
        min_length=1,
        description="索引构建报告文件名",
    )
    skip_existing: bool = Field(
        default=True,
        description="是否跳过已经写入向量库的 chunk",
    )
    fail_on_empty_chunk: bool = Field(
        default=True,
        description="遇到空 chunk 时是否直接失败",
    )

    @model_validator(mode="after")
    def validate_filenames(self) -> IndexBuilderSettings:
        """清理并校验文件名字段。"""

        self.manifest_filename = self.manifest_filename.strip()
        self.build_report_filename = self.build_report_filename.strip()
        if not self.manifest_filename:
            raise ValueError("manifest_filename 不能为空")
        if not self.build_report_filename:
            raise ValueError("build_report_filename 不能为空")
        return self


class IndexingSettings(BaseModel):
    """Embedding、持久化与索引构建配置。"""

    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_repository: VectorRepositorySettings = Field(
        default_factory=VectorRepositorySettings
    )
    builder: IndexBuilderSettings = Field(default_factory=IndexBuilderSettings)
