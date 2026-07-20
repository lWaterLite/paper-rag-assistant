"""文档导入与文档目录 API 契约。"""

from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from app.api.contracts.common import ApiModel, ensure_not_blank


class DocumentIngestRequest(ApiModel):
    """POST /documents/ingest 的请求体。"""

    source_dir: str = Field(description="待导入文档目录")
    rebuild: bool = Field(default=False, description="是否强制重建索引")

    @field_validator("source_dir")
    @classmethod
    def validate_source_dir(cls, value: str) -> str:
        """文档目录不能为空白字符串。

        允许目录白名单依赖运行时 Settings，应由 DocumentSourceAccessService 在
        应用服务层校验，而不是由静态 API 契约直接访问文件系统。
        """

        return ensure_not_blank(value, "source_dir")


class DocumentSummaryResponse(ApiModel):
    """文档列表中的单个文档摘要。"""

    doc_id: str
    version_id: str
    title: str | None = None
    source_path: str
    content_hash: str
    chunk_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentListResponse(ApiModel):
    """GET /documents 的响应体。"""

    documents: list[DocumentSummaryResponse] = Field(default_factory=list)
    total: int


class DocumentIngestResponse(ApiModel):
    """POST /documents/ingest 的响应体。"""

    index_id: str
    document_count: int
    chunk_count: int
    vector_count: int
    manifest: dict[str, Any]
    trace_id: str
