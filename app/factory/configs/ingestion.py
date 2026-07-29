"""Ingestion Settings 到运行时 Config 的适配。"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.settings import IngestionSettings
from app.ingest.chunking.strategies import ChunkerConfig
from app.ingest.loading.access import DocumentSourceAccessConfig
from app.ingest.loading.local import LocalDocumentLoaderConfig
from app.ingest.parsing.cleaners import PdfTextCleanerConfig
from app.ingest.reporting.configuration import (
    ChunkingReportConfig,
    IngestionReportConfig,
)


@dataclass(frozen=True, slots=True)
class IngestionConfigAdapter:
    """将 IngestionSettings 一次转换为可复用的运行时 Config 快照。"""

    settings: IngestionSettings
    loader: LocalDocumentLoaderConfig = field(init=False)
    document_source_access: DocumentSourceAccessConfig = field(init=False)
    pdf_text_cleaner: PdfTextCleanerConfig = field(init=False)
    ingestion_report: IngestionReportConfig = field(init=False)
    chunker: ChunkerConfig = field(init=False)
    chunking_report: ChunkingReportConfig = field(init=False)

    def __post_init__(self) -> None:
        loader_settings = self.settings.loader
        object.__setattr__(
            self,
            "loader",
            LocalDocumentLoaderConfig(
                recursive=loader_settings.recursive,
                ignored_dir_names=loader_settings.ignored_dir_names,
                ignored_relative_paths=loader_settings.ignored_relative_paths,
                skip_hidden_paths=loader_settings.skip_hidden_paths,
                temporary_file_prefixes=loader_settings.temporary_file_prefixes,
                temporary_file_suffixes=loader_settings.temporary_file_suffixes,
            ),
        )
        object.__setattr__(
            self,
            "document_source_access",
            DocumentSourceAccessConfig(
                allowed_source_dirs=self.settings.access.allowed_source_dirs
            ),
        )
        pdf_settings = self.settings.cleaning.pdf
        object.__setattr__(
            self,
            "pdf_text_cleaner",
            PdfTextCleanerConfig(
                edge_line_count=pdf_settings.edge_line_count,
                min_repeat_ratio=pdf_settings.min_repeat_ratio,
                min_line_length=pdf_settings.min_line_length,
                max_line_length=pdf_settings.max_line_length,
            ),
        )
        object.__setattr__(
            self,
            "ingestion_report",
            IngestionReportConfig(output_dir=self.settings.report.output_dir),
        )
        chunking_settings = self.settings.chunking
        object.__setattr__(
            self,
            "chunker",
            ChunkerConfig(
                strategy=chunking_settings.strategy,
                chunk_size=chunking_settings.chunk_size,
                chunk_overlap=chunking_settings.chunk_overlap,
                tokenizer=chunking_settings.tokenizer,
            ),
        )
        object.__setattr__(
            self,
            "chunking_report",
            ChunkingReportConfig(output_dir=chunking_settings.report.output_dir),
        )
