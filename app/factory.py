"""应用对象组装入口。

这个模块是项目的 composition root。底层对象不自己创建依赖，而是在这里统一组装。
这样可以避免同一个进程中出现一部分组件使用 env 配置、一部分组件使用隐式默认配置。
"""

from __future__ import annotations

from app.core.config import Settings
from app.generation.answer_generator import MockAnswerGenerator
from app.indexing.embedding_cache import EmbeddingCache, InMemoryEmbeddingCache
from app.indexing.embeddings import EmbeddingClient, MockEmbeddingClient
from app.indexing.index_builder import IndexBuilder, RagIndex
from app.indexing.vector_store import InMemoryVectorStore
from app.ingest.chunkers import CharacterChunker
from app.ingest.cleaners import BasicTextCleaner, HtmlTextCleaner, PdfTextCleaner
from app.ingest.loaders import (
    DocumentIdentityBuilder,
    LocalDocumentLoader,
    LocalDocumentLoaderConfig,
    LocalTextLoader,
)
from app.ingest.parsers import HtmlDocumentParser, MarkdownParser, ParserRegistry, PdfDocumentParser, PlainTextParser
from app.ingest.pipeline import IngestionPipeline
from app.pipeline import RagPipeline
from app.retrieval.context_packer import SimpleContextPacker
from app.retrieval.retrievers import Retriever, VectorRetriever
from app.storage.repositories import InMemoryDocumentRepository


def build_loader_config(settings: Settings) -> LocalDocumentLoaderConfig:
    """从应用 Settings 转换成本地文档 loader 配置。"""

    return LocalDocumentLoaderConfig(recursive=settings.loader_recursive_iter)


def build_document_identity_builder() -> DocumentIdentityBuilder:
    """创建文档身份生成器。"""

    return DocumentIdentityBuilder()


def build_local_document_loader(settings: Settings) -> LocalDocumentLoader:
    """创建完整 ingestion 使用的本地文档 loader。"""

    return LocalDocumentLoader(
        config=build_loader_config(settings),
        identity_builder=build_document_identity_builder(),
    )


def build_local_text_loader(settings: Settings) -> LocalTextLoader:
    """创建兼容旧文本流程的 loader。"""

    return LocalTextLoader(
        config=build_loader_config(settings),
        identity_builder=build_document_identity_builder(),
    )


def build_parser_registry() -> ParserRegistry:
    """创建文档解析器注册表。"""

    text_cleaner = BasicTextCleaner()
    return ParserRegistry(
        parsers=[
            MarkdownParser(cleaner=text_cleaner),
            HtmlDocumentParser(cleaner=HtmlTextCleaner()),
            PdfDocumentParser(cleaner=PdfTextCleaner()),
            PlainTextParser(cleaner=text_cleaner),
        ]
    )


def build_ingestion_pipeline(settings: Settings) -> IngestionPipeline:
    """创建文档摄取 pipeline。"""

    return IngestionPipeline(
        loader=build_local_document_loader(settings),
        parser_registry=build_parser_registry(),
    )


def build_index_builder(
    settings: Settings,
    *,
    ingestion_pipeline: IngestionPipeline | None = None,
    embedding_client: EmbeddingClient | None = None,
    embedding_cache: EmbeddingCache | None = None,
    vector_store: InMemoryVectorStore | None = None,
    repository: InMemoryDocumentRepository | None = None,
) -> IndexBuilder:
    """创建离线索引构建器。

    这里允许测试或实验显式覆盖某些依赖；生产入口使用默认组装即可。
    """

    return IndexBuilder(
        settings=settings,
        ingestion_pipeline=ingestion_pipeline if ingestion_pipeline is not None else build_ingestion_pipeline(settings),
        chunker=CharacterChunker(settings),
        embedding_client=embedding_client if embedding_client is not None else MockEmbeddingClient(settings),
        embedding_cache=embedding_cache if embedding_cache is not None else InMemoryEmbeddingCache(),
        vector_store=vector_store if vector_store is not None else InMemoryVectorStore(),
        repository=repository if repository is not None else InMemoryDocumentRepository(),
    )


def build_rag_pipeline(
    settings: Settings,
    index: RagIndex,
    *,
    retriever: Retriever | None = None,
    context_packer: SimpleContextPacker | None = None,
    answer_generator: MockAnswerGenerator | None = None,
) -> RagPipeline:
    """创建在线 RAG 问答 pipeline。"""

    return RagPipeline(
        settings=settings,
        retriever=retriever if retriever is not None else VectorRetriever(index.embedding_client, index.vector_store),
        context_packer=context_packer if context_packer is not None else SimpleContextPacker(settings.max_context_chars),
        answer_generator=answer_generator if answer_generator is not None else MockAnswerGenerator(),
    )
