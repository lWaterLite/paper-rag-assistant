"""运行时 Config 快照聚合入口。"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.settings import EnvSettings, ProjectSettings
from app.factory.configs.indexing import IndexingConfigAdapter
from app.factory.configs.ingestion import IngestionConfigAdapter
from app.factory.configs.pipeline import PipelineConfigAdapter
from app.factory.configs.retrieval import RetrievalConfigAdapter
from app.factory.configs.generation import GenerationConfigAdapter
from app.generation.configuration import CitationValidationConfig, GenerationConfig
from app.indexing.configuration import EmbeddingConfig, IndexBuilderConfig, VectorRepositoryConfig
from app.ingest.chunking.strategies import ChunkerConfig
from app.ingest.loading.access import DocumentSourceAccessConfig
from app.ingest.loading.local import LocalDocumentLoaderConfig
from app.ingest.parsing.cleaners import PdfTextCleanerConfig
from app.ingest.reporting.configuration import ChunkingReportConfig, IngestionReportConfig
from app.pipeline import RagPipelineConfig
from app.retrieval.configuration import HybridRetrievalConfig, RetrievalConfig
from app.retrieval.configuration.postprocessing import PostProcessingConfig
from app.retrieval.context import ContextPackerConfig
from app.retrieval.context.evidence_transformers import EvidenceTransformationConfig
from app.retrieval.context.token_estimators import TokenEstimatorConfig
from app.retrieval.rerankers import RerankingConfig
from app.retrieval.reporting import RetrievalReportConfig
from app.retrieval.tokenizers import TokenizerConfig
from app.llm import LlmClientConfig
from app.retrieval.query import QueryPlanningConfig


@dataclass(frozen=True, slots=True)
class ConfigFactory:
    """按应用生命周期缓存 Settings 转换得到的不可变 Config 快照。

    每个 ApplicationFactory 持有独立的 ConfigFactory。它不使用全局缓存，也不支持
    原地热更新；配置变化应通过重建 ApplicationFactory 与 Runtime 生效。
    """

    env_settings: EnvSettings
    project_settings: ProjectSettings
    ingestion: IngestionConfigAdapter = field(init=False)
    indexing: IndexingConfigAdapter = field(init=False)
    retrieval: RetrievalConfigAdapter = field(init=False)
    generation: GenerationConfigAdapter = field(init=False)
    pipeline: PipelineConfigAdapter = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ingestion",
            IngestionConfigAdapter(self.project_settings.ingestion),
        )
        object.__setattr__(
            self,
            "indexing",
            IndexingConfigAdapter(self.project_settings.indexing),
        )
        object.__setattr__(
            self,
            "retrieval",
            RetrievalConfigAdapter(self.project_settings.retrieval),
        )
        object.__setattr__(
            self,
            "generation",
            GenerationConfigAdapter(
                self.project_settings.generation,
                self.project_settings.retrieval,
            ),
        )
        object.__setattr__(
            self,
            "pipeline",
            PipelineConfigAdapter(self.project_settings.retrieval),
        )

    def build_loader_config(self) -> LocalDocumentLoaderConfig:
        """返回缓存的本地文档 Loader Config。"""

        return self.ingestion.loader

    def build_document_source_access_config(self) -> DocumentSourceAccessConfig:
        """返回缓存的受限文档目录访问 Config。"""

        return self.ingestion.document_source_access

    def build_pdf_text_cleaner_config(self) -> PdfTextCleanerConfig:
        """返回缓存的 PDF 清洗 Config。"""

        return self.ingestion.pdf_text_cleaner

    def build_ingestion_report_config(self) -> IngestionReportConfig:
        """返回缓存的摄取报告 Config。"""

        return self.ingestion.ingestion_report

    def build_chunker_config(self) -> ChunkerConfig:
        """返回缓存的 Chunker Config。"""

        return self.ingestion.chunker

    def build_chunking_report_config(self) -> ChunkingReportConfig:
        """返回缓存的 Chunking 报告 Config。"""

        return self.ingestion.chunking_report

    def build_embedding_config(self) -> EmbeddingConfig:
        """返回缓存的 Embedding Config。"""

        return self.indexing.embedding

    def build_vector_repository_config(self) -> VectorRepositoryConfig:
        """返回缓存的向量持久化 Config。"""

        return self.indexing.vector_repository

    def build_index_builder_config(self) -> IndexBuilderConfig:
        """返回缓存的索引构建 Config。"""

        return self.indexing.index_builder

    def build_retrieval_config(self) -> RetrievalConfig:
        """返回缓存的检索 Config。"""

        return self.retrieval.retrieval

    def build_tokenizer_config(self) -> TokenizerConfig:
        """返回缓存的分词器 Config。"""

        return self.retrieval.tokenizer

    def build_hybrid_retrieval_config(self) -> HybridRetrievalConfig:
        """返回缓存的 Hybrid Retrieval Config。"""

        return self.retrieval.hybrid

    def build_reranking_config(self) -> RerankingConfig:
        """返回缓存的 Reranking Config。"""

        return self.retrieval.reranking

    def build_token_estimator_config(self) -> TokenEstimatorConfig:
        """返回缓存的 Token Estimator Config。"""

        return self.retrieval.token_estimator

    def build_context_packer_config(self) -> ContextPackerConfig:
        """返回缓存的 Context Packer Config。"""

        return self.retrieval.context_packer

    def build_evidence_transformation_config(self) -> EvidenceTransformationConfig:
        """返回缓存的证据变换 Config。"""

        return self.retrieval.evidence_transformation

    def build_postprocessing_config(self) -> PostProcessingConfig:
        """返回缓存的检索后处理 Config。"""

        return self.retrieval.postprocessing

    def build_rag_pipeline_config(self) -> RagPipelineConfig:
        """返回缓存的顶层 RAG Pipeline Config。"""

        return self.pipeline.rag_pipeline

    def build_retrieval_report_config(self) -> RetrievalReportConfig:
        """返回缓存的检索报告 Config。"""

        return self.retrieval.report

    def build_llm_client_config(self) -> LlmClientConfig:
        """返回缓存的 LLM Client Config。"""

        return self.generation.llm

    def build_query_planning_config(self) -> QueryPlanningConfig:
        """返回缓存的查询规划 Config。"""

        return self.generation.query_planning

    def build_generation_config(self) -> GenerationConfig:
        """返回缓存的回答生成 Config。"""

        return self.generation.answering

    def build_citation_validation_config(self) -> CitationValidationConfig:
        """返回缓存的 Citation 校验 Config。"""

        return self.generation.citation_validation
