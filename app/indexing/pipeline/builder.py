"""离线索引构建流程。"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

from app.core.errors import AppError, ErrorCode
from app.core.tracing import RagTrace
from app.ingest.chunking.models import DocumentChunk
from app.indexing.configuration import (
    EmbeddingConfig,
    IndexBuilderConfig,
    VectorRepositoryConfig,
)
from app.indexing.embeddings import (
    EmbeddingCache,
    EmbeddingClient,
    validate_embedding_vectors,
)
from app.indexing.manifests import (
    BUILDING_INDEX_STATUS,
    FAILED_INDEX_STATUS,
    IndexManifest,
    IndexVersionStatus,
    READY_INDEX_STATUS,
)
from app.indexing.reporting import IndexBuildReportWriter
from app.indexing.collections import VectorCollection, VectorRecord
from app.ingest.chunking.collection import ChunkCollection
from app.ingest.reporting import (
    ChunkingReportConfig,
    ChunkingReportWriter,
    IngestionReportConfig,
    IngestionReportWriter,
)
from app.ingest.chunking.strategies import Chunker
from app.ingest.collections import DocumentCollection
from app.ingest.models import RawDocument
from app.ingest.pipeline import IngestionPipeline
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.index_manifest_repository import ManifestRepository
from app.repositories.vector_repository import VectorRepository

from app.indexing.pipeline.types import IndexBuildResult, RagIndex


class IndexBuilder:
    """离线索引构建器。"""

    def __init__(
        self,
        *,
        config: IndexBuilderConfig,
        embedding_config: EmbeddingConfig,
        vector_repository_config: VectorRepositoryConfig,
        ingestion_pipeline: IngestionPipeline,
        chunker: Chunker,
        embedding_client: EmbeddingClient,
        embedding_cache: EmbeddingCache,
        vector_collection: VectorCollection,
        document_collection: DocumentCollection,
        chunk_collection: ChunkCollection,
        vector_repository: VectorRepository,
        document_repository: DocumentRepository,
        chunk_repository: ChunkRepository,
        manifest_repository: ManifestRepository,
        build_report_writer: IndexBuildReportWriter,
        ingestion_report_writer: IngestionReportWriter,
        ingestion_report_config: IngestionReportConfig,
        chunking_report_writer: ChunkingReportWriter,
        chunking_report_config: ChunkingReportConfig,
    ) -> None:
        self._config = config
        self._embedding_config = embedding_config
        self._vector_repository_config = vector_repository_config
        self._ingestion_pipeline = ingestion_pipeline
        self._chunker = chunker
        self._embedding_client = embedding_client
        self._embedding_cache = embedding_cache
        self._vector_collection = vector_collection
        self._document_collection = document_collection
        self._chunk_collection = chunk_collection
        self._vector_repository = vector_repository
        self._document_repository = document_repository
        self._chunk_repository = chunk_repository
        self._manifest_repository = manifest_repository
        self._build_report_writer = build_report_writer
        self._ingestion_report_writer = ingestion_report_writer
        self._ingestion_report_config = ingestion_report_config
        self._chunking_report_writer = chunking_report_writer
        self._chunking_report_config = chunking_report_config

    def build_from_directory(
        self, source_dir: Path
    ) -> tuple[RagIndex, IndexBuildResult]:
        """从目录构建内存索引。"""

        self._prepare_output_directories()
        trace = RagTrace()
        building_manifest: IndexManifest | None = None
        latest_chunk_count = 0
        latest_vector_count = 0

        try:
            started = time.perf_counter()
            ingestion_result = self._ingestion_pipeline.ingest_directory(source_dir)
            raw_documents = ingestion_result.raw_documents
            parsed_documents = ingestion_result.parsed_documents
            for document in raw_documents:
                self._document_collection.save_raw(document)
            for document in parsed_documents:
                self._document_collection.save_parsed(document)
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
            building_manifest = self._build_manifest(
                source_dir=source_dir,
                raw_documents=raw_documents,
                chunk_count=0,
                vector_count=0,
                status=BUILDING_INDEX_STATUS,
            )
            building_manifest_path = self._write_manifest(building_manifest)
            self._record_manifest_stage(
                trace=trace,
                stage="manifest_building",
                started=started,
                manifest=building_manifest,
                manifest_path=building_manifest_path,
            )

            started = time.perf_counter()
            all_chunks = []
            for document in parsed_documents:
                chunks = self._chunker.split(document)
                all_chunks.extend(chunks)
            latest_chunk_count = len(all_chunks)
            empty_chunks = [chunk for chunk in all_chunks if not chunk.text.strip()]
            if empty_chunks and self._config.fail_on_empty_chunk:
                raise AppError(
                    ErrorCode.INDEX_FAILED,
                    f"索引构建发现空 chunk：{len(empty_chunks)} 个，请先检查 chunking 或清洗流程",
                    trace_id=trace.trace_id,
                    trace=trace,
                )
            indexable_chunks = [chunk for chunk in all_chunks if chunk.text.strip()]
            self._chunk_collection.add_many(all_chunks)
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
                    chunk
                    for chunk in indexable_chunks
                    if not self._vector_collection.contains_chunk(chunk.chunk_id)
                ]
            else:
                chunks_to_index = indexable_chunks
            skipped_existing_chunks = len(indexable_chunks) - len(chunks_to_index)
            vectors, cache_hits, cache_misses = self._embed_chunks_with_cache(
                chunks_to_index
            )
            for chunk, vector in zip(chunks_to_index, vectors, strict=True):
                self._vector_collection.add(_build_vector_record(chunk, vector))
            latest_vector_count = self._vector_collection.count()
            self._embedding_cache.persist()
            self._vector_repository.save(self._vector_collection)
            self._document_repository.save(self._document_collection)
            self._chunk_repository.save(self._chunk_collection)
            trace.record_stage(
                "indexing",
                "success",
                started,
                {
                    "vector_count": self._vector_collection.count(),
                    "embedding_cache_hits": cache_hits,
                    "embedding_cache_misses": cache_misses,
                    "skipped_existing_chunks": skipped_existing_chunks,
                    "embedding_cache_count": self._embedding_cache.count(),
                },
            )

            started = time.perf_counter()
            manifest = replace(
                building_manifest,
                status=READY_INDEX_STATUS,
                chunk_count=latest_chunk_count,
                vector_count=latest_vector_count,
            )
            manifest_path = self._write_manifest(manifest)
            self._record_manifest_stage(
                trace=trace,
                stage="manifest_ready",
                started=started,
                manifest=manifest,
                manifest_path=manifest_path,
            )
            trace.mark_success()

            index = RagIndex(
                vector_collection=self._vector_collection,
                document_collection=self._document_collection,
                chunk_collection=self._chunk_collection,
                embedding_client=self._embedding_client,
                manifest=manifest,
            )
            result = IndexBuildResult(
                document_count=len(raw_documents),
                chunk_count=len(all_chunks),
                vector_count=self._vector_collection.count(),
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
            build_report_path = self._write_build_report(result)
            result = replace(result, build_report_path=build_report_path)
            return index, result
        except Exception as exc:
            if trace.final_status == "running":
                trace.mark_failed(type(exc).__name__, str(exc))
            self._write_failed_manifest_if_possible(
                building_manifest=building_manifest,
                chunk_count=latest_chunk_count,
                vector_count=latest_vector_count,
            )
            raise

    def _prepare_output_directories(self) -> None:
        """准备索引构建会写入的目录。"""

        self._vector_repository_config.collection_dir.mkdir(parents=True, exist_ok=True)
        self._ingestion_report_config.output_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        self._chunking_report_config.output_path.parent.mkdir(
            parents=True, exist_ok=True
        )

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

        output_path = (
            self._vector_repository_config.collection_dir
            / self._config.build_report_filename
        )
        return self._build_report_writer.write(result, output_path)

    def _build_manifest(
        self,
        *,
        source_dir: Path,
        raw_documents: list[RawDocument],
        chunk_count: int,
        vector_count: int,
        status: IndexVersionStatus,
    ) -> IndexManifest:
        """根据当前构建上下文生成 manifest。"""

        return IndexManifest.build(
            source_dir=source_dir,
            chunker=type(self._chunker).__name__,
            chunk_size=self._chunker.config.chunk_size,
            chunk_overlap=self._chunker.config.chunk_overlap,
            embedding_provider=self._embedding_client.provider,
            embedding_model=self._embedding_client.model_name,
            embedding_dimension=self._embedding_client.dimension,
            embedding_batch_size=self._embedding_config.batch_size,
            vector_repository_type=self._vector_repository_config.repository_type,
            vector_collection_name=self._vector_repository_config.collection_name,
            distance_metric=self._vector_repository_config.distance_metric,
            document_count=len(raw_documents),
            chunk_count=chunk_count,
            vector_count=vector_count,
            document_versions={
                document.doc_id: document.version_id for document in raw_documents
            },
            status=status,
        )

    def _write_manifest(self, manifest: IndexManifest) -> Path:
        """写入当前索引构建状态的 Manifest。"""

        return self._manifest_repository.write(manifest)

    def _record_manifest_stage(
        self,
        *,
        trace: RagTrace,
        stage: str,
        started: float,
        manifest: IndexManifest,
        manifest_path: Path | None,
    ) -> None:
        """记录 manifest 状态写入阶段。"""

        trace.record_stage(
            stage,
            "success",
            started,
            {
                "manifest_path": manifest_path.as_posix()
                if manifest_path is not None
                else None,
                "index_id": manifest.index_id,
                "schema_version": manifest.schema_version,
                "status": manifest.status,
                "config_hash": manifest.config_hash,
                "document_set_hash": manifest.document_set_hash,
            },
        )

    def _write_failed_manifest_if_possible(
        self,
        *,
        building_manifest: IndexManifest | None,
        chunk_count: int,
        vector_count: int,
    ) -> None:
        """构建失败时尽量把 manifest 状态覆盖为 failed。

        失败状态写入不能掩盖原始异常，所以这里会吞掉写 failed manifest 时的异常。
        """

        if building_manifest is None:
            return
        failed_manifest = replace(
            building_manifest,
            status=FAILED_INDEX_STATUS,
            chunk_count=chunk_count,
            vector_count=vector_count,
        )
        try:
            self._manifest_repository.write(failed_manifest)
        except Exception:
            return

    def _embed_chunks_with_cache(
        self, chunks: list[DocumentChunk]
    ) -> tuple[list[list[float]], int, int]:
        """使用缓存批量生成 chunk embedding。

        缓存命中时直接复用向量；未命中时集中批量请求 embedding client，再写回缓存。
        """

        vectors: list[list[float] | None] = [None] * len(chunks)
        missing_indices: list[int] = []
        missing_texts: list[str] = []
        cache_hits = 0

        for index, chunk in enumerate(chunks):
            cached_vector = self._embedding_cache.get(
                self._embedding_client, chunk.text
            )
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


def _build_vector_record(chunk: DocumentChunk, vector: list[float]) -> VectorRecord:
    """从 DocumentChunk 和 embedding 向量构造轻量向量记录。"""

    return VectorRecord(
        chunk_id=chunk.chunk_id,
        vector=vector,
        metadata={
            "doc_id": chunk.doc_id,
            "content_hash": chunk.content_hash,
            "version_id": chunk.version_id,
            "source_path": chunk.source_path,
            "chunk_index": chunk.chunk_index,
            "title": chunk.title,
            "section": chunk.section,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
        },
    )
