"""Settings 到运行时 Config 的适配工厂。"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.settings import EnvSettings, ProjectSettings
from app.indexing.configs import (
    EmbeddingConfig,
    IndexBuilderConfig,
    VectorRepositoryConfig,
)
from app.ingest.chunking.report import ChunkingReportConfig
from app.ingest.chunking.strategies import ChunkerConfig
from app.ingest.cleaners import PdfTextCleanerConfig
from app.ingest.loaders import LocalDocumentLoaderConfig
from app.ingest.pipeline import IngestionReportConfig
from app.retrieval.configs import BM25Config, RetrievalConfig
from app.retrieval.tokenizers import TokenizerConfig


@dataclass(slots=True)
class ConfigFactory:
    """集中完成 Settings 到 Config 的转换。

    Settings 表示外部配置文件结构，Config 表示功能类实际接收的运行时配置。
    这层只做字段映射和配置对象构造，不创建业务对象。
    """

    env_settings: EnvSettings
    project_settings: ProjectSettings

    def build_loader_config(self) -> LocalDocumentLoaderConfig:
        """从结构化 ProjectSettings 转换成本地文档 loader 配置。"""

        settings = self.project_settings.ingestion.loader
        return LocalDocumentLoaderConfig(
            recursive=settings.recursive,
            ignored_dir_names=settings.ignored_dir_names,
            ignored_relative_paths=settings.ignored_relative_paths,
            skip_hidden_paths=settings.skip_hidden_paths,
            temporary_file_prefixes=settings.temporary_file_prefixes,
            temporary_file_suffixes=settings.temporary_file_suffixes,
        )

    def build_pdf_text_cleaner_config(self) -> PdfTextCleanerConfig:
        """从结构化 ProjectSettings 转换成 PDF cleaner 配置。"""

        settings = self.project_settings.ingestion.cleaning.pdf
        return PdfTextCleanerConfig(
            edge_line_count=settings.edge_line_count,
            min_repeat_ratio=settings.min_repeat_ratio,
            min_line_length=settings.min_line_length,
            max_line_length=settings.max_line_length,
        )

    def build_ingestion_report_config(self) -> IngestionReportConfig:
        """从结构化 ProjectSettings 转换成 ingestion report 配置。"""

        return IngestionReportConfig(
            output_dir=self.project_settings.ingestion.report.output_dir
        )

    def build_chunker_config(self) -> ChunkerConfig:
        """从结构化 ProjectSettings 转换成 chunker 配置。"""

        settings = self.project_settings.ingestion.chunking
        return ChunkerConfig(
            strategy=settings.strategy,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            tokenizer=settings.tokenizer,
        )

    def build_chunking_report_config(self) -> ChunkingReportConfig:
        """从结构化 ProjectSettings 转换成 chunking report 配置。"""

        return ChunkingReportConfig(
            output_dir=self.project_settings.ingestion.chunking.report.output_dir
        )

    def build_embedding_config(self) -> EmbeddingConfig:
        """从结构化 ProjectSettings 转换成 embedding 运行时配置。"""

        settings = self.project_settings.indexing.embedding
        return EmbeddingConfig(
            provider=settings.provider,
            model=settings.model,
            dimension=settings.dimension,
            batch_size=settings.batch_size,
            timeout_seconds=settings.timeout_seconds,
            max_retries=settings.max_retries,
            api_key_env_name=settings.api_key_env_name,
        )

    def build_vector_repository_config(self) -> VectorRepositoryConfig:
        """从结构化 ProjectSettings 转换成向量持久化运行时配置。"""

        settings = self.project_settings.indexing.vector_repository
        return VectorRepositoryConfig(
            repository_type=settings.type,
            index_dir=settings.index_dir,
            collection_name=settings.collection_name,
            distance_metric=settings.distance_metric,
            persist=settings.persist,
        )

    def build_index_builder_config(self) -> IndexBuilderConfig:
        """从结构化 ProjectSettings 转换成索引构建运行时配置。"""

        settings = self.project_settings.indexing.builder
        return IndexBuilderConfig(
            manifest_filename=settings.manifest_filename,
            build_report_filename=settings.build_report_filename,
            skip_existing=settings.skip_existing,
            fail_on_empty_chunk=settings.fail_on_empty_chunk,
        )

    def build_retrieval_config(self) -> RetrievalConfig:
        """从 EnvSettings 和 ProjectSettings 转换成检索运行时配置。"""

        settings = self.project_settings.retrieval
        return RetrievalConfig(
            strategy=self.env_settings.retrieval_strategy,
            top_k=self.env_settings.top_k,
            bm25=BM25Config(
                k1=settings.bm25.k1,
                b=settings.bm25.b,
            ),
            deduplicate_by_chunk_id=settings.deduplicate_by_chunk_id,
        )

    def build_tokenizer_config(self) -> TokenizerConfig:
        """从 ProjectSettings 转换成分词器运行时配置。"""

        return TokenizerConfig(
            strategy=self.project_settings.retrieval.tokenizer.strategy,
        )
