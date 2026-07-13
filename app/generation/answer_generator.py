"""回答生成器。

子模块 1 暂时使用 mock 生成器，重点理解 answer generation 的输入输出结构。
"""

from __future__ import annotations

from typing import Protocol

from app.core.models import RagAnswer, RagTrace, RetrievedChunk
from app.generation.prompts import RagAnswerPrompt, build_rag_answer_prompt
from app.retrieval.context import PackedContext


class AnswerGenerator(Protocol):
    """回答生成器协议。"""

    def generate(
        self,
        question: str,
        packed_context: PackedContext,
        retrieved_chunks: list[RetrievedChunk],
        trace: RagTrace,
    ) -> RagAnswer:
        """基于上下文生成最终回答。"""


class MockAnswerGenerator:
    """基于检索上下文生成一个演示回答。"""

    def build_prompt(
        self, question: str, packed_context: PackedContext
    ) -> RagAnswerPrompt:
        """构造后续真实 LLM 可直接使用的 prompt。"""

        return build_rag_answer_prompt(question, packed_context)

    def generate(
        self,
        question: str,
        packed_context: PackedContext,
        retrieved_chunks: list[RetrievedChunk],
        trace: RagTrace,
    ) -> RagAnswer:
        """生成结构化 RAG 回答。"""

        if not packed_context.citations:
            answer = "当前知识库中没有检索到足够相关的资料，因此不能可靠回答这个问题。"
        else:
            first_citation = packed_context.citations[0]
            answer = (
                f"这是一个 mock RAG 回答。针对问题“{question}”，系统检索到了相关资料，"
                f"你可以先查看引用 [{first_citation.citation_id}] 对应的片段。"
                "后续接入真实 LLM 后，回答生成器应只基于检索上下文组织答案。"
            )

        return RagAnswer(
            answer=answer,
            citations=packed_context.citations,
            retrieved_chunks=retrieved_chunks,
            trace_id=trace.trace_id,
            latency_ms=trace.latency_ms,
        )
