"""确定性的本地 LLM Client，用于离线开发与测试。"""

from __future__ import annotations

import json

from app.llm.models import LlmRequest, LlmResponse


class MockLlmClient:
    """根据请求 metadata 返回可预测的结构化响应。

    它只用于保持离线链路可运行，不模拟模型推理能力。真实工程中应通过注册表替换为
    外部或本地模型适配器。
    """

    @property
    def provider_name(self) -> str:
        """返回稳定的实现名称。"""

        return "mock"

    def complete(self, request: LlmRequest) -> LlmResponse:
        """返回符合当前任务输出契约的确定性内容。"""

        task = request.metadata.get("task", "answer_generation")
        if task == "query_rewrite":
            content = json.dumps(
                {
                    "primary_query": request.metadata.get("question", ""),
                    "additional_queries": [],
                    "keywords": [],
                    "hyde_document": None,
                },
                ensure_ascii=False,
            )
        else:
            citation_id = request.metadata.get("primary_citation_id")
            if citation_id:
                answer = (
                    "当前离线 mock 模型不生成新的事实性归纳。"
                    f"请查看已检索证据 [{citation_id}]。"
                )
                citation_ids = [citation_id]
                abstained = False
                abstention_reason = None
            else:
                answer = "根据当前知识库资料无法确定。"
                citation_ids = []
                abstained = True
                abstention_reason = "没有可用的检索证据"
            content = json.dumps(
                {
                    "answer": answer,
                    "citation_ids": citation_ids,
                    "abstained": abstained,
                    "abstention_reason": abstention_reason,
                },
                ensure_ascii=False,
            )

        return LlmResponse(content=content, model=request.model)

