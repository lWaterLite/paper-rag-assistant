"""Settings 到运行时 Config 的适配工厂。"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.settings import EnvSettings, ProjectSettings
from app.indexing.configs import EmbeddingConfig, IndexBuilderConfig, VectorRepositoryConfig
from app.ingest.chunking.report import ChunkingReportConfig
from app.ingest.chunking.strategies import ChunkerConfig
from app.ingest.cleaners import PdfTextCleanerConfig
from app.ingest.loaders import LocalDocumentLoaderConfig
from app.ingest.pipeline import IngestionReportConfig
from app.retrieval.configs import BM25Config, RetrievalConfig


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

        return LocalDocumentLoaderConfig(
            recursive=self.project_settings.loader.recursive,
            ignored_dir_names=self.project_settings.loader.ignored_dir_names,
            ignored_relative_paths=self.project_settings.loader.ignored_relative_paths,
            skip_hidden_paths=self.project_settings.loader.skip_hidden_paths,
            temporary_file_prefixes=self.project_settings.loader.temporary_file_prefixes,
            temporary_file_suffixes=self.project_settings.loader.temporary_file_suffixes,
        )

    def build_pdf_text_cleaner_config(self) -> PdfTextCleanerConfig:
        """从结构化 ProjectSettings 转换成 PDF cleaner 配置。"""

        return PdfTextCleanerConfig(
            edge_line_count=self.project_settings.pdf_cleaner.edge_line_count,
            min_repeat_ratio=self.project_settings.pdf_cleaner.min_repeat_ratio,
            min_line_length=self.project_settings.pdf_cleaner.min_line_length,
            max_line_length=self.project_settings.pdf_cleaner.max_line_length,
        )

    def build_ingestion_report_config(self) -> IngestionReportConfig:
        """从结构化 ProjectSettings 转换成 ingestion report 配置。"""

        return IngestionReportConfig(output_dir=self.project_settings.ingestion_report.output_dir)

    def build_chunker_config(self) -> ChunkerConfig:
        """从结构化 ProjectSettings 转换成 chunker 配置。"""

        return ChunkerConfig(
            strategy=self.project_settings.chunking.strategy,
            chunk_size=self.project_settings.chunking.chunk_size,
            chunk_overlap=self.project_settings.chunking.chunk_overlap,
            tokenizer=self.project_settings.chunking.tokenizer,
        )

    def build_chunking_report_config(self) -> ChunkingReportConfig:
        """从结构化 ProjectSettings 转换成 chunking report 配置。"""

        return ChunkingReportConfig(output_dir=self.project_settings.chunking_report.output_dir)

    def build_embedding_config(self) -> EmbeddingConfig:
        """从结构化 ProjectSettings 转换成 embedding 运行时配置。"""

        return EmbeddingConfig(
            provider=self.project_settings.embedding.provider,
            model=self.project_settings.embedding.model,
            dimension=self.project_settings.embedding.dimension,
            batch_size=self.project_settings.embedding.batch_size,
            timeout_seconds=self.project_settings.embedding.timeout_seconds,
            max_retries=self.project_settings.embedding.max_retries,
            api_key_env_name=self.project_settings.embedding.api_key_env_name,
        )

    def build_vector_repository_config(self) -> VectorRepositoryConfig:
        """从结构化 ProjectSettings 转换成向量持久化运行时配置。"""

        return VectorRepositoryConfig(
            repository_type=self.project_settings.vector_repository.type,
            index_dir=self.project_settings.vector_repository.index_dir,
            collection_name=self.project_settings.vector_repository.collection_name,
            distance_metric=self.project_settings.vector_repository.distance_metric,
            persist=self.project_settings.vector_repository.persist,
        )

    def build_index_builder_config(self) -> IndexBuilderConfig:
        """从结构化 ProjectSettings 转换成索引构建运行时配置。"""

        return IndexBuilderConfig(
            manifest_filename=self.project_settings.index_builder.manifest_filename,
            build_report_filename=self.project_settings.index_builder.build_report_filename,
            skip_existing=self.project_settings.index_builder.skip_existing,
            fail_on_empty_chunk=self.project_settings.index_builder.fail_on_empty_chunk,
        )

    def build_retrieval_config(self) -> RetrievalConfig:
        """从 EnvSettings 和 ProjectSettings 转换成检索运行时配置。"""

        return RetrievalConfig(
            strategy=self.env_settings.retrieval_strategy,
            top_k=self.env_settings.top_k,
            bm25=BM25Config(
                k1=self.project_settings.retrieval.bm25_k1,
                b=self.project_settings.retrieval.bm25_b,
            ),
            deduplicate_by_chunk_id=self.project_settings.retrieval.deduplicate_by_chunk_id,
        )
