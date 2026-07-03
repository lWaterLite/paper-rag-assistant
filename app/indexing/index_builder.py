"""离线索引构建流程。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from app.core.errors import AppError, ErrorCode
from app.core.models import DocumentChunk, RagTrace
from app.indexing.configs import EmbeddingConfig, IndexBuilderConfig, VectorStoreConfig
from app.indexing.embedding_cache import EmbeddingCache
from app.indexing.embeddings import EmbeddingClient, validate_embedding_vectors
from app.indexing.manifest import IndexManifest, IndexManifestStore
from app.indexing.report import IndexBuildReportWriter
from app.indexing.vector_store import VectorStore
from app.ingest.chunking.report import ChunkingReportConfig, ChunkingReportWriter
from app.ingest.chunking.strategies import Chunker
from app.ingest.pipeline import IngestionFailure, IngestionPipeline, IngestionReportConfig, IngestionReportWriter
from app.storage.repositories import InMemoryDocumentRepository


@dataclass(frozen=True)
class IndexBuildResult:
    """一次索引构建的结果摘要。"""

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
    """构建完成后的索引对象。"""

    vector_store: VectorStore
    repository: InMemoryDocumentRepository
    embedding_client: EmbeddingClient
    manifest: IndexManifest


class IndexBuilder:
    """离线索引构建器。"""

    def __init__(
        self,
        *,
        config: IndexBuilderConfig,
        embedding_config: EmbeddingConfig,
        vector_store_config: VectorStoreConfig,
        ingestion_pipeline: IngestionPipeline,
        chunker: Chunker,
        embedding_client: EmbeddingClient,
        embedding_cache: EmbeddingCache,
        vector_store: VectorStore,
        repository: InMemoryDocumentRepository,
        manifest_store: IndexManifestStore,
        build_report_writer: IndexBuildReportWriter,
        ingestion_report_writer: IngestionReportWriter,
        ingestion_report_config: IngestionReportConfig,
        chunking_report_writer: ChunkingReportWriter,
        chunking_report_config: ChunkingReportConfig,
    ) -> None:
        self._config = config
        self._embedding_config = embedding_config
        self._vector_store_config = vector_store_config
        self._ingestion_pipeline = ingestion_pipeline
        self._chunker = chunker
        self._embedding_client = embedding_client
        self._embedding_cache = embedding_cache
        self._vector_store = vector_store
        self._repository = repository
        self._manifest_store = manifest_store
        self._build_report_writer = build_report_writer
        self._ingestion_report_writer = ingestion_report_writer
        self._ingestion_report_config = ingestion_report_config
        self._chunking_report_writer = chunking_report_writer
        self._chunking_report_config = chunking_report_config

    def build_from_directory(self, source_dir: Path) -> tuple[RagIndex, IndexBuildResult]:
        """从目录构建内存索引。"""

        self._prepare_output_directories()
        trace = RagTrace()

        started = time.perf_counter()
        ingestion_result = self._ingestion_pipeline.ingest_directory(source_dir)
        raw_documents = ingestion_result.raw_documents
        parsed_documents = ingestion_result.parsed_documents
        for document in raw_documents:
            self._repository.save_raw(document)
        for document in parsed_documents:
            self._repository.save_parsed(document)
        ingestion_report_output_path = self._prepare_ingestion_report_output()
        ingestion_report_path = self._ingestion_report_writer.write(
            ingestion_result,
            ingestion_report_output_path,
        )
        trace.record_stage(
            "ingestion",
            "success",
            started,
            {
                "document_count": len(parsed_documents),
                "failed_files": len(ingestion_result.failures),
                "report_path": ingestion_report_path.as_posix(),
            },
        )

        started = time.perf_counter()
        all_chunks = []
        for document in parsed_documents:
            chunks = self._chunker.split(document)
            all_chunks.extend(chunks)
        empty_chunks = [chunk for chunk in all_chunks if not chunk.text.strip()]
        if empty_chunks and self._config.fail_on_empty_chunk:
            raise AppError(
                ErrorCode.INDEX_FAILED,
                f"索引构建发现空 chunk：{len(empty_chunks)} 个，请先检查 chunking 或清洗流程",
                trace_id=trace.trace_id,
                trace=trace,
            )
        indexable_chunks = [chunk for chunk in all_chunks if chunk.text.strip()]
        self._repository.save_chunks(all_chunks)
        chunking_report_output_path = self._prepare_chunking_report_output()
        chunking_report_path = self._chunking_report_writer.write(
            documents=parsed_documents,
            chunks=all_chunks,
            config=self._chunker.config,
            output_path=chunking_report_output_path,
        )
        trace.record_stage(
            "chunking",
            "success",
            started,
            {
                "chunk_count": len(all_chunks),
                "empty_chunk_count": len(empty_chunks),
                "report_path": chunking_report_path.as_posix(),
            },
        )

        started = time.perf_counter()
        if self._config.skip_existing:
            chunks_to_index = [
                chunk for chunk in indexable_chunks
                if not self._vector_store.contains_chunk(chunk.chunk_id)
            ]
        else:
            chunks_to_index = indexable_chunks
        skipped_existing_chunks = len(indexable_chunks) - len(chunks_to_index)
        vectors, cache_hits, cache_misses = self._embed_chunks_with_cache(chunks_to_index)
        for chunk, vector in zip(chunks_to_index, vectors, strict=True):
            self._vector_store.add(chunk, vector)
        if self._vector_store_config.persist:
            self._embedding_cache.persist()
            self._vector_store.persist()
        trace.record_stage(
            "indexing",
            "success",
            started,
            {
                "vector_count": self._vector_store.count(),
                "embedding_cache_hits": cache_hits,
                "embedding_cache_misses": cache_misses,
                "skipped_existing_chunks": skipped_existing_chunks,
                "embedding_cache_count": self._embedding_cache.count(),
            },
        )

        started = time.perf_counter()
        manifest = IndexManifest.build(
            source_dir=source_dir,
            chunker=type(self._chunker).__name__,
            chunk_size=self._chunker.config.chunk_size,
            chunk_overlap=self._chunker.config.chunk_overlap,
            embedding_provider=self._embedding_client.provider,
            embedding_model=self._embedding_client.model_name,
            embedding_dimension=self._embedding_client.dimension,
            embedding_batch_size=self._embedding_config.batch_size,
            vector_store_type=self._vector_store_config.store_type,
            vector_collection_name=self._vector_store_config.collection_name,
            distance_metric=self._vector_store_config.distance_metric,
            document_count=len(raw_documents),
            chunk_count=len(all_chunks),
            vector_count=self._vector_store.count(),
            document_versions={document.doc_id: document.version_id for document in raw_documents},
        )
        manifest_path = self._manifest_store.write(manifest) if self._vector_store_config.persist else None
        trace.record_stage(
            "manifest",
            "success",
            started,
            {
                "manifest_path": manifest_path.as_posix() if manifest_path is not None else None,
                "index_id": manifest.index_id,
                "config_hash": manifest.config_hash,
            },
        )
        trace.mark_success()

        index = RagIndex(
            vector_store=self._vector_store,
            repository=self._repository,
            embedding_client=self._embedding_client,
            manifest=manifest,
        )
        result = IndexBuildResult(
            document_count=len(raw_documents),
            chunk_count=len(all_chunks),
            vector_count=self._vector_store.count(),
            manifest=manifest,
            trace=trace,
            embedding_cache_hits=cache_hits,
            embedding_cache_misses=cache_misses,
            skipped_existing_chunks=skipped_existing_chunks,
            empty_chunk_count=len(empty_chunks),
            ingestion_failures=ingestion_result.failures,
            ingestion_report_path=ingestion_report_path,
            chunking_report_path=chunking_report_path,
            manifest_path=manifest_path,
        )
        build_report_path = self._write_build_report(result) if self._vector_store_config.persist else None
        result = replace(result, build_report_path=build_report_path)
        return index, result

    def _prepare_output_directories(self) -> None:
        """准备索引构建会写入的目录。"""

        if self._vector_store_config.persist:
            self._vector_store_config.collection_dir.mkdir(parents=True, exist_ok=True)
        self._ingestion_report_config.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._chunking_report_config.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _prepare_ingestion_report_output(self) -> Path:
        """准备 ingestion 报告输出路径。

        目录创建属于索引构建流程的运行产物准备，不放在 writer 中。
        writer 只负责把报告内容写入已经确定的文件路径。
        """

        output_path = self._ingestion_report_config.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def _prepare_chunking_report_output(self) -> Path:
        """准备 chunking 报告输出路径。"""

        output_path = self._chunking_report_config.output_path
        return output_path

    def _write_build_report(self, result: IndexBuildResult) -> Path:
        """写入索引构建报告。"""

        output_path = self._vector_store_config.collection_dir / self._config.build_report_filename
        return self._build_report_writer.write(result, output_path)

    def _embed_chunks_with_cache(self, chunks: list[DocumentChunk]) -> tuple[list[list[float]], int, int]:
        """使用缓存批量生成 chunk embedding。

        缓存命中时直接复用向量；未命中时集中批量请求 embedding client，再写回缓存。
        """

        vectors: list[list[float] | None] = [None] * len(chunks)
        missing_indices: list[int] = []
        missing_texts: list[str] = []
        cache_hits = 0

        for index, chunk in enumerate(chunks):
            cached_vector = self._embedding_cache.get(self._embedding_client, chunk.text)
            if cached_vector is None:
                missing_indices.append(index)
                missing_texts.append(chunk.text)
                continue
            vectors[index] = cached_vector
            cache_hits += 1

        new_vectors = self._embedding_client.embed_batch(missing_texts)
        validate_embedding_vectors(
            expected_count=len(missing_texts),
            vectors=new_vectors,
            expected_dimension=self._embedding_client.dimension,
            context="embedding client 返回结果",
        )
        for chunk_index, vector in zip(missing_indices, new_vectors, strict=True):
            chunk_text = chunks[chunk_index].text
            self._embedding_cache.set(self._embedding_client, chunk_text, vector)
            vectors[chunk_index] = vector

        resolved_vectors = [vector for vector in vectors if vector is not None]
        if len(resolved_vectors) != len(chunks):
            raise RuntimeError("embedding cache 内部错误：部分 chunk 没有生成向量")
        validate_embedding_vectors(
            expected_count=len(chunks),
            vectors=resolved_vectors,
            expected_dimension=self._embedding_client.dimension,
            context="索引构建 embedding 结果",
        )
        return resolved_vectors, cache_hits, len(missing_indices)
