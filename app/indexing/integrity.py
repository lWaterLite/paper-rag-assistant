"""索引持久化产物的跨集合完整性校验。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import NoReturn

from app.core.errors import AppError, ErrorCode
from app.indexing.collections import VectorCollection, VectorRecord
from app.indexing.manifests import IndexManifest
from app.ingest.chunking.collection import ChunkCollection
from app.ingest.chunking.models import DocumentChunk
from app.ingest.collections import DocumentCollection
from app.ingest.models import ParsedDocument, RawDocument


def validate_index_artifact_integrity(
    *,
    manifest: IndexManifest,
    vector_collection: VectorCollection,
    document_collection: DocumentCollection,
    chunk_collection: ChunkCollection,
) -> None:
    """验证索引版本内各持久化产物的数量与引用关系。

    这是一项基础完整性校验：它只验证实体身份和来源链，不在加载路径中重算
    全文哈希或 embedding，以控制启动成本。更昂贵的内容级诊断应由未来的
    独立完整性检查命令承担。
    """

    raw_documents = tuple(document_collection.iter_raw())
    parsed_documents = tuple(document_collection.iter_parsed())
    chunks = tuple(chunk_collection.iter_chunks())
    vector_records = tuple(vector_collection.iter_records())

    _validate_document_artifacts(
        manifest=manifest,
        raw_documents=raw_documents,
        parsed_documents=parsed_documents,
    )
    _validate_chunk_artifacts(
        manifest=manifest,
        chunks=chunks,
        parsed_documents=parsed_documents,
        collection_count=chunk_collection.count(),
    )
    _validate_vector_artifacts(
        manifest=manifest,
        vector_collection=vector_collection,
        vector_records=vector_records,
        chunks=chunks,
    )


def _validate_document_artifacts(
    *,
    manifest: IndexManifest,
    raw_documents: tuple[RawDocument, ...],
    parsed_documents: tuple[ParsedDocument, ...],
) -> None:
    """验证原始文档、解析文档与 Manifest 的身份映射。"""

    raw_by_id = _index_by_id(
        raw_documents,
        entity_name="原始文档",
        get_id=lambda document: document.doc_id,
    )
    if len(raw_documents) != manifest.document_count:
        _raise_integrity_error(
            "原始文档数量与 manifest 不一致："
            f"manifest={manifest.document_count}，raw_documents={len(raw_documents)}"
        )

    actual_versions = {
        document.doc_id: document.version_id for document in raw_documents
    }
    if actual_versions != manifest.document_versions:
        _raise_integrity_error(
            "原始文档版本映射与 manifest 不一致："
            f"manifest={manifest.document_versions}，actual={actual_versions}"
        )

    for document in parsed_documents:
        raw_document = raw_by_id.get(document.doc_id)
        if raw_document is None:
            _raise_integrity_error(
                f"解析文档引用了不存在的原始文档：doc_id={document.doc_id}"
            )
        _validate_document_identity(
            raw_document=raw_document,
            parsed_document=document,
        )


def _validate_chunk_artifacts(
    *,
    manifest: IndexManifest,
    chunks: tuple[DocumentChunk, ...],
    parsed_documents: tuple[ParsedDocument, ...],
    collection_count: int,
) -> None:
    """验证 Chunk 必须来自当前版本的解析文档。"""

    parsed_by_id = _index_by_id(
        parsed_documents,
        entity_name="解析文档",
        get_id=lambda document: document.doc_id,
    )
    _index_by_id(chunks, entity_name="Chunk", get_id=lambda chunk: chunk.chunk_id)

    if collection_count != len(chunks):
        _raise_integrity_error(
            "Chunk 集合计数与可遍历记录数不一致："
            f"count={collection_count}，chunks={len(chunks)}"
        )
    if len(chunks) != manifest.chunk_count:
        _raise_integrity_error(
            "Chunk 数量与 manifest 不一致："
            f"manifest={manifest.chunk_count}，chunk_collection={len(chunks)}"
        )

    for chunk in chunks:
        parsed_document = parsed_by_id.get(chunk.doc_id)
        if parsed_document is None:
            _raise_integrity_error(
                f"Chunk 引用了不存在的解析文档：chunk_id={chunk.chunk_id}，"
                f"doc_id={chunk.doc_id}"
            )
        _validate_chunk_identity(chunk=chunk, parsed_document=parsed_document)


def _validate_vector_artifacts(
    *,
    manifest: IndexManifest,
    vector_collection: VectorCollection,
    vector_records: tuple[VectorRecord, ...],
    chunks: tuple[DocumentChunk, ...],
) -> None:
    """验证向量记录必须引用当前 Chunk，并镜像其身份 metadata。"""

    if vector_collection.count() != len(vector_records):
        _raise_integrity_error(
            "向量集合计数与可遍历记录数不一致："
            f"count={vector_collection.count()}，records={len(vector_records)}"
        )
    if len(vector_records) != manifest.vector_count:
        _raise_integrity_error(
            "索引向量数量与 manifest 不一致："
            f"manifest={manifest.vector_count}，vector_collection={len(vector_records)}"
        )
    if (
        manifest.vector_count > 0
        and vector_collection.dimension
        != manifest.artifact_definition.runtime_compatibility.embedding.dimension
    ):
        _raise_integrity_error(
            "索引向量维度与 manifest 不一致："
            "manifest="
            f"{manifest.artifact_definition.runtime_compatibility.embedding.dimension}，"
            f"vector_collection={vector_collection.dimension}"
        )

    chunks_by_id = _index_by_id(
        chunks,
        entity_name="Chunk",
        get_id=lambda chunk: chunk.chunk_id,
    )
    _index_by_id(
        vector_records,
        entity_name="向量记录",
        get_id=lambda record: record.chunk_id,
    )

    for record in vector_records:
        chunk = chunks_by_id.get(record.chunk_id)
        if chunk is None:
            _raise_integrity_error(
                f"向量记录引用了不存在的 Chunk：chunk_id={record.chunk_id}"
            )
        _validate_vector_metadata(record=record, chunk=chunk)


def _validate_document_identity(
    *,
    raw_document: RawDocument,
    parsed_document: ParsedDocument,
) -> None:
    """验证解析文档仍代表同一份原始文档版本。"""

    for field, raw_value, parsed_value in (
        ("version_id", raw_document.version_id, parsed_document.version_id),
        ("content_hash", raw_document.content_hash, parsed_document.content_hash),
        ("source_path", raw_document.source_path, parsed_document.source_path),
    ):
        if raw_value != parsed_value:
            _raise_integrity_error(
                f"解析文档与原始文档 {field} 不一致：doc_id={raw_document.doc_id}，"
                f"raw={raw_value!r}，parsed={parsed_value!r}"
            )


def _validate_chunk_identity(
    *,
    chunk: DocumentChunk,
    parsed_document: ParsedDocument,
) -> None:
    """验证 Chunk 的版本、内容身份和来源路径。"""

    for field, chunk_value, document_value in (
        ("version_id", chunk.version_id, parsed_document.version_id),
        ("content_hash", chunk.content_hash, parsed_document.content_hash),
        ("source_path", chunk.source_path, parsed_document.source_path),
    ):
        if chunk_value != document_value:
            _raise_integrity_error(
                f"Chunk 与解析文档 {field} 不一致：chunk_id={chunk.chunk_id}，"
                f"chunk={chunk_value!r}，document={document_value!r}"
            )


def _validate_vector_metadata(*, record: VectorRecord, chunk: DocumentChunk) -> None:
    """验证向量 metadata 与 Chunk 的轻量身份字段保持一致。"""

    expected_metadata = {
        "doc_id": chunk.doc_id,
        "content_hash": chunk.content_hash,
        "version_id": chunk.version_id,
        "source_path": chunk.source_path,
        "chunk_index": chunk.chunk_index,
    }
    for field, expected_value in expected_metadata.items():
        actual_value = record.metadata.get(field)
        if actual_value != expected_value:
            _raise_integrity_error(
                f"向量记录 metadata 与 Chunk 不一致：chunk_id={record.chunk_id}，"
                f"field={field}，vector={actual_value!r}，chunk={expected_value!r}"
            )


def _index_by_id[T](
    entities: Iterable[T],
    *,
    entity_name: str,
    get_id: Callable[[T], str],
) -> dict[str, T]:
    """以稳定 ID 建立映射，并拒绝重复实体。"""

    indexed: dict[str, T] = {}
    for entity in entities:
        entity_id = str(get_id(entity))
        if entity_id in indexed:
            _raise_integrity_error(f"{entity_name} 存在重复 ID：{entity_id}")
        indexed[entity_id] = entity
    return indexed


def _raise_integrity_error(message: str) -> NoReturn:
    """以统一错误码报告索引产物不完整。"""

    raise AppError(ErrorCode.INDEX_FAILED, f"索引产物完整性校验失败：{message}")
