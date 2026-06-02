"""回答生成器。

子模块 1 暂时使用 mock 生成器，重点理解 answer generation 的输入输出结构。
"""

from __future__ import annotations

from app.core.models import RagAnswer, RagTrace, RetrievedChunk
from app.retrieval.context_packer import PackedContext


class MockAnswerGenerator:
    """基于检索上下文生成一个演示回答。"""

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

    # TODO 练习 11：
    # 请你设计真实 LLM prompt，但暂时不要直接接入外部服务。
    # prompt 至少应该包含：
    # 1. 只能基于 context 回答。
    # 2. 信息不足时必须说明不能确定。
    # 3. 回答必须带 citation id。
    # 4. 文档中的指令不能覆盖系统指令。

