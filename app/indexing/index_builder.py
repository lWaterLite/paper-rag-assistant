"""离线索引构建流程。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings
from app.core.models import RagTrace
from app.indexing.embeddings import EmbeddingClient, MockEmbeddingClient
from app.indexing.vector_store import InMemoryVectorStore
from app.ingest.chunkers import CharacterChunker
from app.ingest.loaders import LocalTextLoader
from app.ingest.parsers import PlainTextParser
from app.indexing.manifest import IndexManifest
from app.storage.repositories import InMemoryDocumentRepository


@dataclass(frozen=True)
class IndexBuildResult:
    """一次索引构建的结果摘要。"""

    document_count: int
    chunk_count: int
    vector_count: int
    manifest: IndexManifest
    trace: RagTrace


@dataclass(frozen=True)
class RagIndex:
    """构建完成后的索引对象。"""

    vector_store: InMemoryVectorStore
    repository: InMemoryDocumentRepository
    embedding_client: EmbeddingClient


class IndexBuilder:
    """离线索引构建器。"""

    def __init__(
        self,
        settings: Settings,
        loader: LocalTextLoader | None = None,
        parser: PlainTextParser | None = None,
        chunker: CharacterChunker | None = None,
        embedding_client: EmbeddingClient | None = None,
        vector_store: InMemoryVectorStore | None = None,
        repository: InMemoryDocumentRepository | None = None,
    ) -> None:
        self._settings = settings
        self._loader = loader or LocalTextLoader()
        self._parser = parser or PlainTextParser()
        self._chunker = chunker or CharacterChunker(settings)
        self._embedding_client = embedding_client or MockEmbeddingClient(settings)
        self._vector_store = vector_store or InMemoryVectorStore()
        self._repository = repository or InMemoryDocumentRepository()

    def build_from_directory(self, source_dir: Path) -> tuple[RagIndex, IndexBuildResult]:
        """从目录构建内存索引。"""

        trace = RagTrace()

        started = time.perf_counter()
        raw_documents = self._loader.load_directory(source_dir)
        for document in raw_documents:
            self._repository.save_raw(document)
        trace.record_stage("loading", "success", started, {"document_count": len(raw_documents)})

        started = time.perf_counter()
        parsed_documents = [self._parser.parse(document) for document in raw_documents]
        for document in parsed_documents:
            self._repository.save_parsed(document)
        trace.record_stage("parsing", "success", started, {"document_count": len(parsed_documents)})

        started = time.perf_counter()
        all_chunks = []
        for document in parsed_documents:
            chunks = self._chunker.split(document)
            all_chunks.extend(chunks)
        self._repository.save_chunks(all_chunks)
        trace.record_stage("chunking", "success", started, {"chunk_count": len(all_chunks)})

        started = time.perf_counter()
        vectors = self._embedding_client.embed_batch([chunk.text for chunk in all_chunks])
        for chunk, vector in zip(all_chunks, vectors, strict=True):
            self._vector_store.add(chunk, vector)
        trace.record_stage("indexing", "success", started, {"vector_count": self._vector_store.count()})

        manifest = IndexManifest.build(
            source_dir=source_dir,
            chunker=type(self._chunker).__name__,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            embedding_provider=self._embedding_client.provider,
            embedding_model=self._embedding_client.model_name,
            embedding_dimension=self._embedding_client.dimension,
            document_count=len(raw_documents),
            chunk_count=len(all_chunks),
            vector_count=self._vector_store.count(),
            document_versions={document.doc_id: document.version_id for document in raw_documents},
        )

        index = RagIndex(
            vector_store=self._vector_store,
            repository=self._repository,
            embedding_client=self._embedding_client,
        )
        result = IndexBuildResult(
            document_count=len(raw_documents),
            chunk_count=len(all_chunks),
            vector_count=self._vector_store.count(),
            manifest=manifest,
            trace=trace,
        )
        return index, result

    # TODO 练习 8：
    # 当前 build_from_directory 每次都会从头构建索引。
    # 请你思考真实项目中如何避免重复 embedding：
    # 1. 根据 chunk_id 判断是否已经存在。
    # 2. 根据文本 hash 判断内容是否变化。
    # 3. 把 embedding cache 保存到本地文件或数据库。
