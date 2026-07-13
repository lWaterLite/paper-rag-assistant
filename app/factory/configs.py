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
from app.pipeline import RagPipelineConfig
from app.retrieval.context_packer import ContextPackerConfig
from app.retrieval.configs import (
    BM25Config,
    HybridRetrievalConfig,
    RetrievalConfig,
)
from app.retrieval.rerankers import RerankingConfig
from app.retrieval.reporting import RetrievalReportConfig
from app.retrieval.token_estimators import TokenEstimatorConfig
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
        """从 ProjectSettings 转换成检索运行时配置。"""

        settings = self.project_settings.retrieval
        return RetrievalConfig(
            strategy=settings.strategy,
            top_k=settings.top_k,
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

    def build_hybrid_retrieval_config(self) -> HybridRetrievalConfig:
        """从 ProjectSettings 转换成 hybrid retrieval 运行时配置。"""

        settings = self.project_settings.retrieval.hybrid
        return HybridRetrievalConfig(
            candidate_multiplier=settings.candidate_multiplier,
            rrf_rank_constant=settings.rrf_rank_constant,
            vector_weight=settings.vector_weight,
            bm25_weight=settings.bm25_weight,
        )

    def build_reranking_config(self) -> RerankingConfig:
        """从 ProjectSettings 转换成 rerank 阶段运行时配置。"""

        settings = self.project_settings.retrieval.reranking
        return RerankingConfig(
            enabled=settings.enabled,
            strategy=settings.strategy,
            candidate_limit=settings.candidate_limit,
            batch_size=settings.batch_size,
            failure_mode=settings.failure_mode,
        )

    def build_token_estimator_config(self) -> TokenEstimatorConfig:
        """从 ProjectSettings 转换成模型上下文 token 估算器配置。"""

        return TokenEstimatorConfig(
            strategy=(
                self.project_settings.retrieval.context_packing.token_estimator.strategy
            )
        )

    def build_context_packer_config(self) -> ContextPackerConfig:
        """从 ProjectSettings 转换成上下文组织器运行时配置。"""

        settings = self.project_settings.retrieval.context_packing
        return ContextPackerConfig(
            model_context_window=settings.model_context_window,
            max_context_tokens=settings.max_context_tokens,
            reserved_prompt_tokens=settings.reserved_prompt_tokens,
            reserved_output_tokens=settings.reserved_output_tokens,
            safety_margin_tokens=settings.safety_margin_tokens,
            max_chunks_per_document=settings.max_chunks_per_document,
        )

    def build_rag_pipeline_config(self) -> RagPipelineConfig:
        """从 ProjectSettings 转换成在线 RAG pipeline 配置。"""

        return RagPipelineConfig(top_k=self.project_settings.retrieval.top_k)

    def build_retrieval_report_config(self) -> RetrievalReportConfig:
        """从 ProjectSettings 转换成 retrieval 报告运行时配置。"""

        settings = self.project_settings.retrieval.report
        return RetrievalReportConfig(
            enabled=settings.enabled,
            output_dir=settings.output_dir,
            include_result_text=settings.include_result_text,
            result_preview_chars=settings.result_preview_chars,
            fail_on_write_error=settings.fail_on_write_error,
        )
