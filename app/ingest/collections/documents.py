"""文档运行时集合。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from app.ingest.models import ParsedDocument, RawDocument


class DocumentCollection(Protocol):
    """RawDocument 和 ParsedDocument 的运行时集合协议。"""

    def save_raw(self, document: RawDocument) -> None:
        """保存原始文档。"""

    def save_parsed(self, document: ParsedDocument) -> None:
        """保存解析后的文档。"""

    def get_raw(self, doc_id: str) -> RawDocument | None:
        """根据 doc_id 读取原始文档。"""

    def get_parsed(self, doc_id: str) -> ParsedDocument | None:
        """根据 doc_id 读取解析后的文档。"""

    def iter_raw(self) -> Iterable[RawDocument]:
        """遍历原始文档。"""

    def iter_parsed(self) -> Iterable[ParsedDocument]:
        """遍历解析后的文档。"""

    def stats(self) -> dict[str, int]:
        """返回集合统计信息。"""


class InMemoryDocumentCollection:
    """内存文档集合。"""

    def __init__(self) -> None:
        self.raw_documents: dict[str, RawDocument] = {}
        self.parsed_documents: dict[str, ParsedDocument] = {}

    def save_raw(self, document: RawDocument) -> None:
        """保存原始文档。"""

        self.raw_documents[document.doc_id] = document

    def save_parsed(self, document: ParsedDocument) -> None:
        """保存解析后的文档。"""

        self.parsed_documents[document.doc_id] = document

    def get_raw(self, doc_id: str) -> RawDocument | None:
        """根据 doc_id 读取原始文档。"""

        return self.raw_documents.get(doc_id)

    def get_parsed(self, doc_id: str) -> ParsedDocument | None:
        """根据 doc_id 读取解析后的文档。"""

        return self.parsed_documents.get(doc_id)

    def iter_raw(self) -> Iterable[RawDocument]:
        """遍历原始文档。"""

        return self.raw_documents.values()

    def iter_parsed(self) -> Iterable[ParsedDocument]:
        """遍历解析后的文档。"""

        return self.parsed_documents.values()

    def stats(self) -> dict[str, int]:
        """返回集合统计信息。"""

        return {
            "raw_documents": len(self.raw_documents),
            "parsed_documents": len(self.parsed_documents),
        }
