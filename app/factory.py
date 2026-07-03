"""应用对象组装入口。

这个模块是项目的 composition root。底层对象不自己创建依赖，而是在这里统一组装。
这样可以避免同一个进程中出现一部分组件使用 env 配置、一部分组件使用隐式默认配置。
"""

from __future__ import annotations

from app.core.settings import EnvSettings, ProjectSettings
from app.generation.answer_generator import MockAnswerGenerator
from app.indexing.configs import EmbeddingConfig, IndexBuilderConfig, VectorStoreConfig
from app.indexing.embedding_cache import EmbeddingCache, FileEmbeddingCache, InMemoryEmbeddingCache
from app.indexing.embeddings import EmbeddingClient, MockEmbeddingClient, OpenAIEmbeddingClient
from app.indexing.index_builder import IndexBuilder, RagIndex
from app.indexing.manifest import IndexManifestStore
from app.indexing.report import IndexBuildReportWriter
from app.indexing.vector_store import InMemoryVectorStore, LocalJsonVectorStore, VectorStore
from app.ingest.chunking.registry import ChunkerRegistry, build_default_chunker_registry
from app.ingest.chunking.report import ChunkingReportConfig, ChunkingReportWriter
from app.ingest.chunking.strategies import Chunker, ChunkerConfig
from app.ingest.cleaners import BasicTextCleaner, HtmlTextCleaner, PdfTextCleaner, PdfTextCleanerConfig
from app.ingest.loaders import (
    DocumentIdentityBuilder,
    LocalDocumentLoader,
    LocalDocumentLoaderConfig,
    LocalTextLoader,
)
from app.ingest.parsers import HtmlDocumentParser, MarkdownParser, ParserRegistry, PdfDocumentParser, PlainTextParser
from app.ingest.pipeline import IngestionPipeline, IngestionReportConfig, IngestionReportWriter
from app.pipeline import RagPipeline
from app.retrieval.context_packer import SimpleContextPacker
from app.retrieval.retrievers import Retriever, VectorRetriever
from app.storage.repositories import InMemoryDocumentRepository


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


def build_vector_store_config(project_settings: ProjectSettings) -> VectorStoreConfig:
    """从结构化 ProjectSettings 转换成向量存储运行时配置。"""

    return VectorStoreConfig(
        store_type=project_settings.vector_store.type,
        index_dir=project_settings.vector_store.index_dir,
        collection_name=project_settings.vector_store.collection_name,
        distance_metric=project_settings.vector_store.distance_metric,
        persist=project_settings.vector_store.persist,
    )


def build_index_builder_config(project_settings: ProjectSettings) -> IndexBuilderConfig:
    """从结构化 ProjectSettings 转换成索引构建运行时配置。"""

    return IndexBuilderConfig(
        manifest_filename=project_settings.index_builder.manifest_filename,
        build_report_filename=project_settings.index_builder.build_report_filename,
        skip_existing=project_settings.index_builder.skip_existing,
        fail_on_empty_chunk=project_settings.index_builder.fail_on_empty_chunk,
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

    vector_store_config = build_vector_store_config(project_settings)
    if vector_store_config.store_type == "local_json" and vector_store_config.persist:
        return FileEmbeddingCache(vector_store_config.embedding_cache_path)
    return InMemoryEmbeddingCache()


def build_vector_store(project_settings: ProjectSettings) -> VectorStore:
    """根据配置创建向量存储。"""

    config = build_vector_store_config(project_settings)
    if config.store_type == "memory":
        return InMemoryVectorStore()
    if config.store_type == "local_json":
        return LocalJsonVectorStore(config.vector_store_path)
    raise ValueError(f"不支持的 vector store 类型：{config.store_type}")


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


def build_local_text_loader(project_settings: ProjectSettings) -> LocalTextLoader:
    """创建兼容旧文本流程的 loader。"""

    return LocalTextLoader(
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
        vector_store: VectorStore | None = None,
        repository: InMemoryDocumentRepository | None = None,
        ingestion_report_writer: IngestionReportWriter | None = None,
        chunking_report_writer: ChunkingReportWriter | None = None,
        chunker_registry: ChunkerRegistry | None = None,
) -> IndexBuilder:
    """创建离线索引构建器。

    这里允许测试或实验显式覆盖某些依赖；生产入口使用默认组装即可。
    """

    _ = env_settings
    embedding_config = build_embedding_config(project_settings)
    vector_store_config = build_vector_store_config(project_settings)
    index_builder_config = build_index_builder_config(project_settings)
    return IndexBuilder(
        config=index_builder_config,
        embedding_config=embedding_config,
        vector_store_config=vector_store_config,
        ingestion_pipeline=ingestion_pipeline if ingestion_pipeline is not None else build_ingestion_pipeline(project_settings),
        chunker=build_configured_chunker(project_settings, chunker_registry=chunker_registry),
        embedding_client=embedding_client if embedding_client is not None else build_embedding_client(project_settings),
        embedding_cache=embedding_cache if embedding_cache is not None else build_embedding_cache(project_settings),
        vector_store=vector_store if vector_store is not None else build_vector_store(project_settings),
        repository=repository if repository is not None else InMemoryDocumentRepository(),
        manifest_store=IndexManifestStore(vector_store_config.collection_dir, index_builder_config),
        build_report_writer=IndexBuildReportWriter(),
        ingestion_report_writer=ingestion_report_writer if ingestion_report_writer is not None else IngestionReportWriter(),
        ingestion_report_config=build_ingestion_report_config(project_settings),
        chunking_report_writer=chunking_report_writer if chunking_report_writer is not None else ChunkingReportWriter(),
        chunking_report_config=build_chunking_report_config(project_settings),
    )


def build_rag_pipeline(
        env_settings: EnvSettings,
        index: RagIndex,
        *,
        retriever: Retriever | None = None,
        context_packer: SimpleContextPacker | None = None,
        answer_generator: MockAnswerGenerator | None = None,
) -> RagPipeline:
    """创建在线 RAG 问答 pipeline。"""

    return RagPipeline(
        settings=env_settings,
        retriever=retriever if retriever is not None else VectorRetriever(index.embedding_client, index.vector_store),
        context_packer=context_packer if context_packer is not None else SimpleContextPacker(
            env_settings.max_context_chars),
        answer_generator=answer_generator if answer_generator is not None else MockAnswerGenerator(),
    )
