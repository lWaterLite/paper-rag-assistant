"""Ingestion 相关对象组装。"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.factory.configs import ConfigFactory
from app.ingest.chunking.registry import ChunkerRegistry, build_default_chunker_registry
from app.ingest.chunking.strategies import Chunker
from app.ingest.loading import DocumentIdentityBuilder, LocalDocumentLoader
from app.ingest.parsing import (
    BasicTextCleaner,
    HtmlDocumentParser,
    HtmlTextCleaner,
    MarkdownParser,
    ParserRegistry,
    PdfDocumentParser,
    PdfTextCleaner,
    PlainTextParser,
)
from app.ingest.pipeline import IngestionPipeline
from app.ingest.reporting import (
    ChunkingReportConfig,
    ChunkingReportWriter,
    IngestionReportConfig,
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
    ) -> Chunker:
        """根据项目配置创建 chunker。

        默认使用工厂持有的 registry；调用方也可以显式传入已经注册过外部策略的 registry。
        """

        registry = (
            chunker_registry if chunker_registry is not None else self.chunker_registry
        )
        return registry.create(self.configs.build_chunker_config())

    def build_document_identity_builder(self) -> DocumentIdentityBuilder:
        """创建文档身份生成器。"""

        return DocumentIdentityBuilder()

    def build_local_document_loader(self) -> LocalDocumentLoader:
        """创建完整 ingestion 使用的本地文档 loader。"""

        return LocalDocumentLoader(
            config=self.configs.build_loader_config(),
            identity_builder=self.build_document_identity_builder(),
        )

    def build_pdf_text_cleaner(self) -> PdfTextCleaner:
        """创建 PDF 文本清洗器。"""

        return PdfTextCleaner(config=self.configs.build_pdf_text_cleaner_config())

    def build_parser_registry(self) -> ParserRegistry:
        """创建文档解析器注册表。"""

        text_cleaner = BasicTextCleaner()
        return ParserRegistry(
            parsers=[
                MarkdownParser(cleaner=text_cleaner),
                HtmlDocumentParser(cleaner=HtmlTextCleaner()),
                PdfDocumentParser(cleaner=self.build_pdf_text_cleaner()),
                PlainTextParser(cleaner=text_cleaner),
            ]
        )

    def build_ingestion_pipeline(self) -> IngestionPipeline:
        """创建文档摄取 pipeline。"""

        return IngestionPipeline(
            loader=self.build_local_document_loader(),
            parser_registry=self.build_parser_registry(),
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

        return IngestionIndexingDependencies(
            pipeline=self.build_ingestion_pipeline(),
            chunker=self.build_configured_chunker(),
            ingestion_report_writer=self.build_ingestion_report_writer(),
            ingestion_report_config=self.configs.build_ingestion_report_config(),
            chunking_report_writer=self.build_chunking_report_writer(),
            chunking_report_config=self.configs.build_chunking_report_config(),
        )
