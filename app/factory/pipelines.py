"""在线 pipeline 相关对象组装。"""

from __future__ import annotations

from dataclasses import dataclass

from app.factory.configs import ConfigFactory
from app.factory.retrieval import RetrievalFactory
from app.generation.answer_generator import AnswerGenerator, MockAnswerGenerator
from app.indexing.index_builder import RagIndex
from app.pipeline import RagPipeline
from app.retrieval.context_packer import ContextPacker, SimpleContextPacker
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

        return RagPipeline(
            config=self.configs.build_rag_pipeline_config(),
            retriever=retriever
            if retriever is not None
            else self.retrieval.build_retriever(
                index,
                registry=retriever_registry,
            ),
            context_packer=context_packer
            if context_packer is not None
            else SimpleContextPacker(self.configs.build_context_packer_config()),
            answer_generator=answer_generator
            if answer_generator is not None
            else MockAnswerGenerator(),
        )
