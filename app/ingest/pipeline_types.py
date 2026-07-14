"""文档摄取流程的结果模型。"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.tracing import RagTrace
from app.ingest.models import ParsedDocument, RawDocument


@dataclass(frozen=True)
class IngestionFailure:
    """单个文件在摄取流程中的失败记录。"""

    source_path: str
    stage: str
    error_code: str
    error_message: str


@dataclass(frozen=True)
class IngestedDocument:
    """同一份文档的加载结果和解析结果。"""

    raw_document: RawDocument
    parsed_document: ParsedDocument


@dataclass(frozen=True)
class IngestionResult:
    """一次目录摄取的完整结果。"""

    documents: list[IngestedDocument]
    failures: list[IngestionFailure]
    trace: RagTrace
    metadata: dict[str, int | str] = field(default_factory=dict)

    @property
    def raw_documents(self) -> list[RawDocument]:
        """返回成功摄取文档的原始数据。"""

        return [document.raw_document for document in self.documents]

    @property
    def parsed_documents(self) -> list[ParsedDocument]:
        """返回成功摄取文档的结构化数据。"""

        return [document.parsed_document for document in self.documents]
