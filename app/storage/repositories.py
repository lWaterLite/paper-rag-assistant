"""内存仓储。

当前只用于练习数据流，后续可以替换成 SQLite、PostgreSQL 或文件持久化。
"""

from __future__ import annotations

from collections.abc import Iterable

from app.core.models import DocumentChunk, ParsedDocument, RawDocument


class InMemoryDocumentRepository:
    """保存文档和 chunk 的内存仓储。"""

    def __init__(self) -> None:
        self.raw_documents: dict[str, RawDocument] = {}
        self.parsed_documents: dict[str, ParsedDocument] = {}
        self.chunks: dict[str, DocumentChunk] = {}

    def save_raw(self, document: RawDocument) -> None:
        self.raw_documents[document.doc_id] = document

    def save_parsed(self, document: ParsedDocument) -> None:
        self.parsed_documents[document.doc_id] = document

    def save_chunks(self, chunks: list[DocumentChunk]) -> None:
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk

    def iter_chunks(self) -> Iterable[DocumentChunk]:
        """返回当前仓储中的所有 chunk。

        检索器只需要知道“有一批 DocumentChunk 可以遍历”，不应该依赖具体仓储类型。
        """

        return self.chunks.values()

    def stats(self) -> dict[str, int]:
        return {
            "raw_documents": len(self.raw_documents),
            "parsed_documents": len(self.parsed_documents),
            "chunks": len(self.chunks),
        }

    # TODO 练习 7：
    # 当前仓储只能存在内存里，程序结束后数据会丢失。
    # 请你设计一个 manifest JSON 的结构，用来记录一次索引构建的配置和统计信息。
