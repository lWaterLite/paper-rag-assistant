"""文档集合持久化 Repository。"""

from __future__ import annotations

import base64
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from app.core.models import (
    ParsedBlock,
    ParsedDocument,
    ParseIssue,
    RawDocument,
)
from app.ingest.document_collection import (
    DocumentCollection,
    InMemoryDocumentCollection,
)


class DocumentRepository(Protocol):
    """文档集合持久化协议。"""

    def load(self) -> DocumentCollection:
        """加载文档集合。"""

    def save(self, collection: DocumentCollection) -> None:
        """保存文档集合。"""


class LocalJsonDocumentRepository:
    """基于本地 JSON 文件的文档集合持久化。"""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> DocumentCollection:
        """从 JSON 文件加载文档集合。"""

        collection = InMemoryDocumentCollection()
        if not self._path.exists():
            return collection

        payload = json.loads(self._path.read_text(encoding="utf-8"))
        for item in payload.get("raw_documents", []):
            collection.save_raw(_deserialize_raw_document(item))
        for item in payload.get("parsed_documents", []):
            collection.save_parsed(_deserialize_parsed_document(item))
        return collection

    def save(self, collection: DocumentCollection) -> None:
        """把文档集合保存为 JSON 文件。"""

        payload = {
            "raw_documents": [
                _serialize_raw_document(document)
                for document in sorted(
                    collection.iter_raw(), key=lambda item: item.doc_id
                )
            ],
            "parsed_documents": [
                _serialize_parsed_document(document)
                for document in sorted(
                    collection.iter_parsed(), key=lambda item: item.doc_id
                )
            ],
        }
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _serialize_raw_document(document: RawDocument) -> dict[str, Any]:
    """把 RawDocument 转换为 JSON 友好的 dict。"""

    data = asdict(document)
    raw_bytes = data.pop("raw_bytes")
    data["raw_bytes_base64"] = (
        base64.b64encode(raw_bytes).decode("ascii") if raw_bytes is not None else None
    )
    return data


def _deserialize_raw_document(data: dict[str, Any]) -> RawDocument:
    """从 JSON dict 恢复 RawDocument。"""

    raw_bytes_base64 = data.pop("raw_bytes_base64", None)
    raw_bytes = (
        base64.b64decode(raw_bytes_base64.encode("ascii")) if raw_bytes_base64 else None
    )
    return RawDocument(raw_bytes=raw_bytes, **data)


def _serialize_parsed_document(document: ParsedDocument) -> dict[str, Any]:
    """把 ParsedDocument 转换为 JSON 友好的 dict。"""

    return asdict(document)


def _deserialize_parsed_document(data: dict[str, Any]) -> ParsedDocument:
    """从 JSON dict 恢复 ParsedDocument。"""

    blocks = [ParsedBlock(**item) for item in data.pop("blocks", [])]
    parse_issues = [ParseIssue(**item) for item in data.pop("parse_issues", [])]
    return ParsedDocument(blocks=blocks, parse_issues=parse_issues, **data)
