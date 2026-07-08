"""Ingestion 相关对象组装。"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.factory.configs import ConfigFactory
from app.ingest.chunking.registry import ChunkerRegistry, build_default_chunker_registry
from app.ingest.chunking.strategies import Chunker
from app.ingest.cleaners import BasicTextCleaner, HtmlTextCleaner, PdfTextCleaner
from app.ingest.loaders import DocumentIdentityBuilder, LocalDocumentLoader
from app.ingest.parsers import HtmlDocumentParser, MarkdownParser, ParserRegistry, PdfDocumentParser, PlainTextParser
from app.ingest.pipeline import IngestionPipeline


@dataclass(slots=True)
class IngestionFactory:
    """组装文档加载、解析、清洗和切分相关对象。"""

    configs: ConfigFactory
    chunker_registry: ChunkerRegistry = field(default_factory=build_default_chunker_registry)

    def build_configured_chunker(
            self,
            *,
            chunker_registry: ChunkerRegistry | None = None,
    ) -> Chunker:
        """根据项目配置创建 chunker。

        默认使用工厂持有的 registry；调用方也可以显式传入已经注册过外部策略的 registry。
        """

        registry = chunker_registry if chunker_registry is not None else self.chunker_registry
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
