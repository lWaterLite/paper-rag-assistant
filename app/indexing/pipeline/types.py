"""索引构建与加载流程共享的数据边界。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.core.tracing import RagTrace
from app.indexing.collections.vector import VectorCollection
from app.indexing.embeddings.base import EmbeddingClient
from app.indexing.manifests.models import IndexManifest
from app.ingest.chunking.collection import ChunkCollection
from app.ingest.collections import DocumentCollection
from app.ingest.pipeline_types import IngestionFailure


@dataclass(frozen=True)
class IndexBuildResult:
    """一次离线索引构建的结果摘要。"""

    document_count: int
    chunk_count: int
    vector_count: int
    manifest: IndexManifest
    trace: RagTrace
    embedding_cache_hits: int = 0
    embedding_cache_misses: int = 0
    skipped_existing_chunks: int = 0
    empty_chunk_count: int = 0
    ingestion_failures: list[IngestionFailure] = field(default_factory=list)
    ingestion_report_path: Path | None = None
    chunking_report_path: Path | None = None
    manifest_path: Path | None = None
    build_report_path: Path | None = None


@dataclass(frozen=True)
class RagIndex:
    """可供在线检索流程使用的已加载索引。"""

    vector_collection: VectorCollection
    document_collection: DocumentCollection
    chunk_collection: ChunkCollection
    embedding_client: EmbeddingClient
    manifest: IndexManifest
