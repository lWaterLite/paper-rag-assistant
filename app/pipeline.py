"""RAG pipeline 编排。

这里把在线问答流程串起来：检索 -> 上下文组织 -> 生成。
"""

from __future__ import annotations

import time

from app.core.config import Settings
from app.core.models import RagAnswer, RagTrace
from app.generation.answer_generator import MockAnswerGenerator
from app.indexing.index_builder import RagIndex
from app.retrieval.context_packer import SimpleContextPacker
from app.retrieval.retrievers import VectorRetriever


class RagPipeline:
    """在线 RAG 问答 pipeline。"""

    def __init__(self, settings: Settings, index: RagIndex) -> None:
        self._settings = settings
        self._retriever = VectorRetriever(index.embedding_client, index.vector_store)
        self._context_packer = SimpleContextPacker(settings.max_context_chars)
        self._answer_generator = MockAnswerGenerator()

    def ask(self, question: str) -> RagAnswer:
        """根据用户问题执行一次 RAG 问答。"""

        trace = RagTrace()

        started = time.perf_counter()
        retrieved_chunks = self._retriever.retrieve(question, top_k=self._settings.top_k)
        trace.record_stage(
            "retrieval",
            "success",
            started,
            {
                "query": question,
                "top_k": self._settings.top_k,
                "returned": len(retrieved_chunks),
            },
        )

        started = time.perf_counter()
        packed_context = self._context_packer.pack(retrieved_chunks)
        trace.record_stage(
            "context_packing",
            "success",
            started,
            {
                "used_chunks": len(packed_context.used_chunks),
                "dropped_chunks": len(packed_context.dropped_chunks),
                "citation_count": len(packed_context.citations),
                "context_chars": len(packed_context.context_text),
            },
        )

        started = time.perf_counter()
        answer = self._answer_generator.generate(
            question=question,
            packed_context=packed_context,
            retrieved_chunks=retrieved_chunks,
            trace=trace,
        )
        trace.record_stage("generation", "success", started, {"answer_chars": len(answer.answer)})

        return answer

    # TODO 练习 12：
    # 当前 pipeline 没有异常捕获和失败 trace。
    # 请你为 retrieval、context_packing、generation 三个阶段补充错误处理，
    # 让失败时也能记录 final_status 和 failure_type。
