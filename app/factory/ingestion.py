"""Ingestion 相关对象组装。"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.factory.configs import ConfigFactory
from app.ingest.chunking.registry import ChunkerRegistry, build_default_chunker_registry
from app.ingest.chunking.strategies import Chunker, ChunkerConfig
from app.ingest.loading.access import (
    DocumentSourceAccessConfig,
    DocumentSourceAccessService,
)
from app.ingest.loading.local import (
    DocumentIdentityBuilder,
    LocalDocumentLoader,
    LocalDocumentLoaderConfig,
)
from app.ingest.parsing.cleaners import (
    BasicTextCleaner,
    HtmlTextCleaner,
    PdfTextCleaner,
    PdfTextCleanerConfig,
)
from app.ingest.parsing.parsers import (
    HtmlDocumentParser,
    MarkdownParser,
    ParserRegistry,
    PdfDocumentParser,
    PlainTextParser,
)
from app.ingest.pipeline import IngestionPipeline
from app.ingest.reporting.chunking import ChunkingReportWriter
from app.ingest.reporting.configuration import (
    ChunkingReportConfig,
    IngestionReportConfig,
)
from app.ingest.reporting.ingestion import (
    IngestionReportWriter,
)


@dataclass(frozen=True, slots=True)
class IngestionIndexingDependencies:
    """索引构建流程所需的 ingest 组件集合。

    该对象只存在于组合层，避免 IndexingFactory 依赖 ingest 的内部实现细节。
    """

    pipeline: IngestionPipeline
    chunker: Chunker
    ingestion_report_writer: IngestionReportWriter
    ingestion_report_config: IngestionReportConfig
    chunking_report_writer: ChunkingReportWriter
    chunking_report_config: ChunkingReportConfig


@dataclass(slots=True)
class IngestionFactory:
    """组装文档加载、解析、清洗和切分相关对象。"""

    configs: ConfigFactory
    chunker_registry: ChunkerRegistry = field(
        default_factory=build_default_chunker_registry
    )

    def build_configured_chunker(
        self,
        *,
        chunker_registry: ChunkerRegistry | None = None,
        config: ChunkerConfig | None = None,
    ) -> Chunker:
        """根据项目配置创建 chunker。

        默认使用工厂持有的 registry；调用方也可以显式传入已经注册过外部策略的 registry。
        """

        registry = (
            chunker_registry if chunker_registry is not None else self.chunker_registry
        )
        active_config = config if config is not None else self.configs.ingestion.chunker
        return registry.create(active_config)

    def build_document_identity_builder(self) -> DocumentIdentityBuilder:
        """创建文档身份生成器。"""

        return DocumentIdentityBuilder()

    def build_document_source_access_service(
        self,
        *,
        config: DocumentSourceAccessConfig | None = None,
    ) -> DocumentSourceAccessService:
        """创建 API 等受限入口使用的文档目录访问服务。"""

        active_config = (
            config
            if config is not None
            else self.configs.ingestion.document_source_access
        )
        return DocumentSourceAccessService(config=active_config)

    def build_local_document_loader(
        self,
        *,
        config: LocalDocumentLoaderConfig | None = None,
    ) -> LocalDocumentLoader:
        """创建完整 ingestion 使用的本地文档 loader。"""

        active_config = config if config is not None else self.configs.ingestion.loader
        return LocalDocumentLoader(
            config=active_config,
            identity_builder=self.build_document_identity_builder(),
        )

    def build_pdf_text_cleaner(
        self,
        *,
        config: PdfTextCleanerConfig | None = None,
    ) -> PdfTextCleaner:
        """创建 PDF 文本清洗器。"""

        active_config = (
            config if config is not None else self.configs.ingestion.pdf_text_cleaner
        )
        return PdfTextCleaner(config=active_config)

    def build_parser_registry(
        self,
        *,
        pdf_text_cleaner_config: PdfTextCleanerConfig | None = None,
    ) -> ParserRegistry:
        """创建文档解析器注册表。"""

        text_cleaner = BasicTextCleaner()
        return ParserRegistry(
            parsers=[
                MarkdownParser(cleaner=text_cleaner),
                HtmlDocumentParser(cleaner=HtmlTextCleaner()),
                PdfDocumentParser(
                    cleaner=self.build_pdf_text_cleaner(config=pdf_text_cleaner_config)
                ),
                PlainTextParser(cleaner=text_cleaner),
            ]
        )

    def build_ingestion_pipeline(
        self,
        *,
        loader_config: LocalDocumentLoaderConfig | None = None,
        pdf_text_cleaner_config: PdfTextCleanerConfig | None = None,
    ) -> IngestionPipeline:
        """创建文档摄取 pipeline。"""

        return IngestionPipeline(
            loader=self.build_local_document_loader(config=loader_config),
            parser_registry=self.build_parser_registry(
                pdf_text_cleaner_config=pdf_text_cleaner_config
            ),
        )

    @staticmethod
    def build_ingestion_report_writer() -> IngestionReportWriter:
        """创建摄取报告写入器。"""

        return IngestionReportWriter()

    @staticmethod
    def build_chunking_report_writer() -> ChunkingReportWriter:
        """创建切分质量报告写入器。"""

        return ChunkingReportWriter()

    def build_indexing_dependencies(self) -> IngestionIndexingDependencies:
        """组装索引构建所需要的 ingest 组件。"""

        config = self.configs.ingestion
        return IngestionIndexingDependencies(
            pipeline=self.build_ingestion_pipeline(
                loader_config=config.loader,
                pdf_text_cleaner_config=config.pdf_text_cleaner,
            ),
            chunker=self.build_configured_chunker(config=config.chunker),
            ingestion_report_writer=self.build_ingestion_report_writer(),
            ingestion_report_config=config.ingestion_report,
            chunking_report_writer=self.build_chunking_report_writer(),
            chunking_report_config=config.chunking_report,
        )
