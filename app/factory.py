"""应用对象组装入口。

这个模块是项目的 composition root。底层对象不自己创建依赖，而是在这里统一组装。
这样可以避免同一个进程中出现一部分组件使用 env 配置、一部分组件使用隐式默认配置。
"""

from __future__ import annotations

from app.core.settings import EnvSettings, ProjectSettings
from app.generation.answer_generator import MockAnswerGenerator
from app.indexing.embedding_cache import EmbeddingCache, InMemoryEmbeddingCache
from app.indexing.embeddings import EmbeddingClient, MockEmbeddingClient
from app.indexing.index_builder import IndexBuilder, RagIndex
from app.indexing.vector_store import InMemoryVectorStore
from app.ingest.chunkers import Chunker, ChunkerConfig, build_chunker
from app.ingest.chunking_report import ChunkingReportConfig, ChunkingReportWriter
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


def build_configured_chunker(project_settings: ProjectSettings) -> Chunker:
    """根据项目配置创建 chunker。"""

    return build_chunker(build_chunker_config(project_settings))


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
        vector_store: InMemoryVectorStore | None = None,
        repository: InMemoryDocumentRepository | None = None,
        ingestion_report_writer: IngestionReportWriter | None = None,
        chunking_report_writer: ChunkingReportWriter | None = None,
) -> IndexBuilder:
    """创建离线索引构建器。

    这里允许测试或实验显式覆盖某些依赖；生产入口使用默认组装即可。
    """

    return IndexBuilder(
        settings=env_settings,
        ingestion_pipeline=ingestion_pipeline if ingestion_pipeline is not None else build_ingestion_pipeline(project_settings),
        chunker=build_configured_chunker(project_settings),
        embedding_client=embedding_client if embedding_client is not None else MockEmbeddingClient(env_settings),
        embedding_cache=embedding_cache if embedding_cache is not None else InMemoryEmbeddingCache(),
        vector_store=vector_store if vector_store is not None else InMemoryVectorStore(),
        repository=repository if repository is not None else InMemoryDocumentRepository(),
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
