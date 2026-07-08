"""应用对象组装入口。

这个模块是项目的 composition root。底层对象不自己创建依赖，而是在这里统一组装。
这样可以避免同一个进程中出现一部分组件使用 env 配置、一部分组件使用隐式默认配置。
"""

from __future__ import annotations

from app.core.errors import AppError, ErrorCode
from app.core.settings import EnvSettings, ProjectSettings
from app.generation.answer_generator import MockAnswerGenerator
from app.indexing.configs import EmbeddingConfig, IndexBuilderConfig, VectorRepositoryConfig
from app.indexing.embedding_cache import EmbeddingCache, FileEmbeddingCache, InMemoryEmbeddingCache
from app.indexing.embeddings import EmbeddingClient, MockEmbeddingClient, OpenAIEmbeddingClient
from app.indexing.index_builder import IndexBuilder, RagIndex
from app.indexing.index_loader import validate_index_from_storage
from app.indexing.report import IndexBuildReportWriter
from app.indexing.vector_collection import InMemoryVectorCollection, VectorCollection
from app.ingest.chunking.collection import ChunkCollection, InMemoryChunkCollection
from app.ingest.chunking.registry import ChunkerRegistry, build_default_chunker_registry
from app.ingest.chunking.report import ChunkingReportConfig, ChunkingReportWriter
from app.ingest.chunking.strategies import Chunker, ChunkerConfig
from app.ingest.cleaners import BasicTextCleaner, HtmlTextCleaner, PdfTextCleaner, PdfTextCleanerConfig
from app.ingest.loaders import (
    DocumentIdentityBuilder,
    LocalDocumentLoader,
    LocalDocumentLoaderConfig,
)
from app.ingest.document_collection import DocumentCollection, InMemoryDocumentCollection
from app.ingest.parsers import HtmlDocumentParser, MarkdownParser, ParserRegistry, PdfDocumentParser, PlainTextParser
from app.ingest.pipeline import IngestionPipeline, IngestionReportConfig, IngestionReportWriter
from app.pipeline import RagPipeline
from app.repositories.chunk_repository import ChunkRepository, LocalJsonChunkRepository
from app.repositories.document_repository import DocumentRepository, LocalJsonDocumentRepository
from app.repositories.index_manifest_repository import IndexManifestRepository
from app.repositories.vector_repository import LocalJsonVectorRepository, VectorRepository
from app.retrieval.configs import BM25Config, RetrievalConfig
from app.retrieval.context_packer import SimpleContextPacker
from app.retrieval.retrievers import BM25Retriever, Retriever, VectorRetriever
from app.retrieval.service import SearchService


def build_loader_config(project_settings: ProjectSettings) -> LocalDocumentLoaderConfig:
    """从结构化 ProjectSettings 转换成本地文档 loader 配置。"""

    return LocalDocumentLoaderConfig(
        recursive=project_settings.loader.recursive,
        ignored_dir_names=project_settings.loader.ignored_dir_names,
        ignored_relative_paths=project_settings.loader.ignored_relative_paths,
        skip_hidden_paths=project_settings.loader.skip_hidden_paths,
        temporary_file_prefixes=project_settings.loader.temporary_file_prefixes,
        temporary_file_suffixes=project_settings.loader.temporary_file_suffixes,
    )


def build_pdf_text_cleaner_config(project_settings: ProjectSettings) -> PdfTextCleanerConfig:
    """从结构化 ProjectSettings 转换成 PDF cleaner 配置。"""

    return PdfTextCleanerConfig(
        edge_line_count=project_settings.pdf_cleaner.edge_line_count,
        min_repeat_ratio=project_settings.pdf_cleaner.min_repeat_ratio,
        min_line_length=project_settings.pdf_cleaner.min_line_length,
        max_line_length=project_settings.pdf_cleaner.max_line_length,
    )


def build_ingestion_report_config(project_settings: ProjectSettings) -> IngestionReportConfig:
    """从结构化 ProjectSettings 转换成 ingestion report 配置。"""

    return IngestionReportConfig(output_dir=project_settings.ingestion_report.output_dir)


def build_chunker_config(project_settings: ProjectSettings) -> ChunkerConfig:
    """从结构化 ProjectSettings 转换成 chunker 配置。"""

    return ChunkerConfig(
        strategy=project_settings.chunking.strategy,
        chunk_size=project_settings.chunking.chunk_size,
        chunk_overlap=project_settings.chunking.chunk_overlap,
        tokenizer=project_settings.chunking.tokenizer,
    )


def build_chunking_report_config(project_settings: ProjectSettings) -> ChunkingReportConfig:
    """从结构化 ProjectSettings 转换成 chunking report 配置。"""

    return ChunkingReportConfig(output_dir=project_settings.chunking_report.output_dir)


def build_embedding_config(project_settings: ProjectSettings) -> EmbeddingConfig:
    """从结构化 ProjectSettings 转换成 embedding 运行时配置。"""

    return EmbeddingConfig(
        provider=project_settings.embedding.provider,
        model=project_settings.embedding.model,
        dimension=project_settings.embedding.dimension,
        batch_size=project_settings.embedding.batch_size,
        timeout_seconds=project_settings.embedding.timeout_seconds,
        max_retries=project_settings.embedding.max_retries,
        api_key_env_name=project_settings.embedding.api_key_env_name,
    )


def build_vector_repository_config(project_settings: ProjectSettings) -> VectorRepositoryConfig:
    """从结构化 ProjectSettings 转换成向量持久化运行时配置。"""

    return VectorRepositoryConfig(
        repository_type=project_settings.vector_repository.type,
        index_dir=project_settings.vector_repository.index_dir,
        collection_name=project_settings.vector_repository.collection_name,
        distance_metric=project_settings.vector_repository.distance_metric,
        persist=project_settings.vector_repository.persist,
    )


def build_index_builder_config(project_settings: ProjectSettings) -> IndexBuilderConfig:
    """从结构化 ProjectSettings 转换成索引构建运行时配置。"""

    return IndexBuilderConfig(
        manifest_filename=project_settings.index_builder.manifest_filename,
        build_report_filename=project_settings.index_builder.build_report_filename,
        skip_existing=project_settings.index_builder.skip_existing,
        fail_on_empty_chunk=project_settings.index_builder.fail_on_empty_chunk,
    )


def build_retrieval_config(env_settings: EnvSettings, project_settings: ProjectSettings) -> RetrievalConfig:
    """从 EnvSettings 和 ProjectSettings 转换成检索运行时配置。"""

    return RetrievalConfig(
        strategy=env_settings.retrieval_strategy,
        top_k=env_settings.top_k,
        bm25=BM25Config(
            k1=project_settings.retrieval.bm25_k1,
            b=project_settings.retrieval.bm25_b,
        ),
        deduplicate_by_chunk_id=project_settings.retrieval.deduplicate_by_chunk_id,
    )


def build_embedding_client(project_settings: ProjectSettings) -> EmbeddingClient:
    """根据配置创建 embedding client。"""

    config = build_embedding_config(project_settings)
    if config.provider == "mock":
        return MockEmbeddingClient(config)
    if config.provider == "openai":
        return OpenAIEmbeddingClient(config)
    raise ValueError(f"不支持的 embedding provider：{config.provider}")


def build_embedding_cache(project_settings: ProjectSettings) -> EmbeddingCache:
    """根据配置创建 embedding cache。"""

    vector_repository_config = build_vector_repository_config(project_settings)
    if vector_repository_config.repository_type == "local_json" and vector_repository_config.persist:
        return FileEmbeddingCache(vector_repository_config.embedding_cache_path)
    return InMemoryEmbeddingCache()


def build_vector_collection(project_settings: ProjectSettings) -> VectorCollection:
    """创建空的向量运行时集合。"""

    _ = project_settings
    return InMemoryVectorCollection()


def build_vector_repository(project_settings: ProjectSettings) -> VectorRepository:
    """根据配置创建向量集合持久化 Repository。"""

    config = build_vector_repository_config(project_settings)
    if config.repository_type in {"memory", "local_json"}:
        return LocalJsonVectorRepository(config.vector_collection_path)
    raise ValueError(f"不支持的 vector repository 类型：{config.repository_type}")


def build_document_collection() -> DocumentCollection:
    """创建文档运行时集合。"""

    return InMemoryDocumentCollection()


def build_chunk_collection() -> ChunkCollection:
    """创建 chunk 运行时集合。"""

    return InMemoryChunkCollection()


def build_document_repository(project_settings: ProjectSettings) -> DocumentRepository:
    """根据配置创建文档集合持久化 Repository。"""

    config = build_vector_repository_config(project_settings)
    return LocalJsonDocumentRepository(config.document_collection_path)


def build_chunk_repository(project_settings: ProjectSettings) -> ChunkRepository:
    """根据配置创建 chunk 集合持久化 Repository。"""

    config = build_vector_repository_config(project_settings)
    return LocalJsonChunkRepository(config.chunk_collection_path)


def build_configured_chunker(
        project_settings: ProjectSettings,
        *,
        chunker_registry: ChunkerRegistry | None = None,
) -> Chunker:
    """根据项目配置创建 chunker。

    默认使用项目内置 registry；调用方也可以传入已经注册过外部策略的 registry。
    """

    registry = chunker_registry if chunker_registry is not None else build_default_chunker_registry()
    return registry.create(build_chunker_config(project_settings))


def build_document_identity_builder() -> DocumentIdentityBuilder:
    """创建文档身份生成器。"""

    return DocumentIdentityBuilder()


def build_local_document_loader(project_settings: ProjectSettings) -> LocalDocumentLoader:
    """创建完整 ingestion 使用的本地文档 loader。"""

    return LocalDocumentLoader(
        config=build_loader_config(project_settings),
        identity_builder=build_document_identity_builder(),
    )


def build_pdf_text_cleaner(project_settings: ProjectSettings) -> PdfTextCleaner:
    """创建 PDF 文本清洗器。"""

    return PdfTextCleaner(config=build_pdf_text_cleaner_config(project_settings))


def build_parser_registry(project_settings: ProjectSettings) -> ParserRegistry:
    """创建文档解析器注册表。"""

    text_cleaner = BasicTextCleaner()
    return ParserRegistry(
        parsers=[
            MarkdownParser(cleaner=text_cleaner),
            HtmlDocumentParser(cleaner=HtmlTextCleaner()),
            PdfDocumentParser(cleaner=build_pdf_text_cleaner(project_settings)),
            PlainTextParser(cleaner=text_cleaner),
        ]
    )


def build_ingestion_pipeline(project_settings: ProjectSettings) -> IngestionPipeline:
    """创建文档摄取 pipeline。"""

    return IngestionPipeline(
        loader=build_local_document_loader(project_settings),
        parser_registry=build_parser_registry(project_settings),
    )


def build_index_builder(
        env_settings: EnvSettings,
        project_settings: ProjectSettings,
        *,
        ingestion_pipeline: IngestionPipeline | None = None,
        embedding_client: EmbeddingClient | None = None,
        embedding_cache: EmbeddingCache | None = None,
        vector_collection: VectorCollection | None = None,
        document_collection: DocumentCollection | None = None,
        chunk_collection: ChunkCollection | None = None,
        vector_repository: VectorRepository | None = None,
        document_repository: DocumentRepository | None = None,
        chunk_repository: ChunkRepository | None = None,
        ingestion_report_writer: IngestionReportWriter | None = None,
        chunking_report_writer: ChunkingReportWriter | None = None,
        chunker_registry: ChunkerRegistry | None = None,
) -> IndexBuilder:
    """创建离线索引构建器。

    这里允许测试或实验显式覆盖某些依赖；生产入口使用默认组装即可。
    """

    _ = env_settings
    embedding_config = build_embedding_config(project_settings)
    vector_repository_config = build_vector_repository_config(project_settings)
    index_builder_config = build_index_builder_config(project_settings)
    return IndexBuilder(
        config=index_builder_config,
        embedding_config=embedding_config,
        vector_repository_config=vector_repository_config,
        ingestion_pipeline=ingestion_pipeline if ingestion_pipeline is not None else build_ingestion_pipeline(project_settings),
        chunker=build_configured_chunker(project_settings, chunker_registry=chunker_registry),
        embedding_client=embedding_client if embedding_client is not None else build_embedding_client(project_settings),
        embedding_cache=embedding_cache if embedding_cache is not None else build_embedding_cache(project_settings),
        vector_collection=vector_collection if vector_collection is not None else build_vector_collection(project_settings),
        document_collection=document_collection if document_collection is not None else build_document_collection(),
        chunk_collection=chunk_collection if chunk_collection is not None else build_chunk_collection(),
        vector_repository=vector_repository if vector_repository is not None else build_vector_repository(project_settings),
        document_repository=document_repository if document_repository is not None else build_document_repository(project_settings),
        chunk_repository=chunk_repository if chunk_repository is not None else build_chunk_repository(project_settings),
        manifest_repository=IndexManifestRepository(vector_repository_config.collection_dir, index_builder_config),
        build_report_writer=IndexBuildReportWriter(),
        ingestion_report_writer=ingestion_report_writer if ingestion_report_writer is not None else IngestionReportWriter(),
        ingestion_report_config=build_ingestion_report_config(project_settings),
        chunking_report_writer=chunking_report_writer if chunking_report_writer is not None else ChunkingReportWriter(),
        chunking_report_config=build_chunking_report_config(project_settings),
    )


def build_rag_index_from_storage(project_settings: ProjectSettings) -> RagIndex:
    """从已有持久化索引加载在线 RAG 索引。"""

    vector_repository_config = build_vector_repository_config(project_settings)
    if vector_repository_config.repository_type != "local_json" or not vector_repository_config.persist:
        raise AppError(
            ErrorCode.INVALID_CONFIG,
            "加载已有索引要求 vector_repository.type='local_json' 且 persist=true；"
            "memory 或未持久化配置没有可恢复的索引产物",
        )

    embedding_config = build_embedding_config(project_settings)
    vector_repository = build_vector_repository(project_settings)
    document_repository = build_document_repository(project_settings)
    chunk_repository = build_chunk_repository(project_settings)
    vector_collection = vector_repository.load()
    manifest = IndexManifestRepository(
        vector_repository_config.collection_dir,
        build_index_builder_config(project_settings),
    ).read()
    validate_index_from_storage(
        manifest=manifest,
        embedding_config=embedding_config,
        vector_repository_config=vector_repository_config,
        vector_collection=vector_collection,
    )
    return RagIndex(
        vector_collection=vector_collection,
        document_collection=document_repository.load(),
        chunk_collection=chunk_repository.load(),
        embedding_client=build_embedding_client(project_settings),
        manifest=manifest,
    )


def build_vector_retriever(index: RagIndex) -> VectorRetriever:
    """创建向量检索器。"""

    return VectorRetriever(
        index.embedding_client,
        index.vector_collection,
        index.chunk_collection,
    )


def build_bm25_retriever(index: RagIndex, project_settings: ProjectSettings) -> BM25Retriever:
    """根据当前 chunk collection 创建 BM25 检索器。"""

    config = BM25Config(
        k1=project_settings.retrieval.bm25_k1,
        b=project_settings.retrieval.bm25_b,
    )
    return BM25Retriever(index.chunk_collection.iter_chunks(), config=config)


def build_retriever(
        env_settings: EnvSettings,
        project_settings: ProjectSettings,
        index: RagIndex,
) -> Retriever:
    """根据检索策略创建在线问答默认检索器。"""

    retrieval_config = build_retrieval_config(env_settings, project_settings)
    if retrieval_config.strategy == "vector":
        return build_vector_retriever(index)
    if retrieval_config.strategy == "bm25":
        return build_bm25_retriever(index, project_settings)
    raise AppError(ErrorCode.INVALID_CONFIG, "hybrid 检索将在后续子模块实现，当前请使用 vector 或 bm25")


def build_search_service(
        env_settings: EnvSettings,
        project_settings: ProjectSettings,
        index: RagIndex,
) -> SearchService:
    """创建只执行检索的 SearchService。"""

    return SearchService(
        retrievers={
            "vector": build_vector_retriever(index),
            "bm25": build_bm25_retriever(index, project_settings),
        },
        config=build_retrieval_config(env_settings, project_settings),
    )


def build_rag_pipeline(
        env_settings: EnvSettings,
        index: RagIndex,
        *,
        project_settings: ProjectSettings | None = None,
        retriever: Retriever | None = None,
        context_packer: SimpleContextPacker | None = None,
        answer_generator: MockAnswerGenerator | None = None,
) -> RagPipeline:
    """创建在线 RAG 问答 pipeline。"""

    resolved_project_settings = project_settings if project_settings is not None else ProjectSettings()
    return RagPipeline(
        settings=env_settings,
        retriever=retriever if retriever is not None else build_retriever(
            env_settings,
            resolved_project_settings,
            index,
        ),
        context_packer=context_packer if context_packer is not None else SimpleContextPacker(
            env_settings.max_context_chars),
        answer_generator=answer_generator if answer_generator is not None else MockAnswerGenerator(),
    )
