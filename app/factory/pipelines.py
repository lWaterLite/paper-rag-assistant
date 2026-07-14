"""在线 pipeline 相关对象组装。"""

from __future__ import annotations

from dataclasses import dataclass

from app.factory.configs import ConfigFactory
from app.factory.retrieval import RetrievalFactory
from app.generation.answer_generator import AnswerGenerator, MockAnswerGenerator
from app.indexing.index_builder import RagIndex
from app.pipeline import RagPipeline
from app.retrieval.context import ContextPacker
from app.retrieval.retrievers import Retriever, RetrieverRegistry


@dataclass(slots=True)
class PipelineFactory:
    """组装在线问答 pipeline。"""

    configs: ConfigFactory
    retrieval: RetrievalFactory

    def build_rag_pipeline(
        self,
        index: RagIndex,
        *,
        retriever: Retriever | None = None,
        retriever_registry: RetrieverRegistry | None = None,
        context_packer: ContextPacker | None = None,
        answer_generator: AnswerGenerator | None = None,
    ) -> RagPipeline:
        """创建在线 RAG 问答 pipeline。"""

        if retriever is not None and retriever_registry is not None:
            raise ValueError("retriever 与 retriever_registry 不能同时传入")

        active_registry = retriever_registry
        if retriever is not None:
            configured_retriever = retriever
            strategy = self.configs.build_retrieval_config().strategy
            active_registry = RetrieverRegistry()
            active_registry.register(strategy, lambda: configured_retriever)

        postprocessing_config = self.retrieval.build_postprocessing_config()
        return RagPipeline(
            config=self.configs.build_rag_pipeline_config(),
            retrieval_service=self.retrieval.build_search_service(
                index,
                registry=active_registry,
                postprocessing_config=postprocessing_config,
            ),
            context_packer=context_packer
            if context_packer is not None
            else self.retrieval.build_context_packer(
                postprocessing_config=postprocessing_config,
            ),
            evidence_transform_stage=self.retrieval.build_evidence_transform_stage(
                postprocessing_config=postprocessing_config,
            ),
            answer_generator=answer_generator
            if answer_generator is not None
            else MockAnswerGenerator(),
        )
