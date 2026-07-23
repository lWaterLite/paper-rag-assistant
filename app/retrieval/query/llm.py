"""由 LLM 产生 QueryPlan 的实现。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.llm import LlmClient, LlmMessage, LlmRequest
from app.retrieval.query.config import QueryPlanningConfig
from app.retrieval.query.models import QueryPlan


@dataclass(frozen=True, slots=True)
class LlmQueryPlanner:
    """将 LLM 输出解析为受约束的 QueryPlan。"""

    config: QueryPlanningConfig
    llm_client: LlmClient
    model: str
    timeout_seconds: float
    max_output_tokens: int = 320

    def plan(self, question: str) -> QueryPlan:
        """调用 LLM，并拒绝不符合结构契约的输出。"""

        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("用户问题不能为空")
        response = self.llm_client.complete(
            LlmRequest(
                messages=(
                    LlmMessage(role="system", content=_SYSTEM_PROMPT),
                    LlmMessage(role="user", content=_build_user_prompt(cleaned_question, self.config)),
                ),
                model=self.model,
                temperature=0.0,
                max_output_tokens=self.max_output_tokens,
                timeout_seconds=self.timeout_seconds,
                metadata={"task": "query_rewrite", "question": cleaned_question},
            )
        )
        payload = _parse_payload(response.content)
        additional_queries = tuple(payload.get("additional_queries", ()))
        if not self.config.multi_query_enabled:
            additional_queries = ()
        else:
            additional_queries = additional_queries[: self.config.max_additional_queries]
        hyde_document = payload.get("hyde_document") if self.config.hyde_enabled else None
        return QueryPlan(
            original_query=cleaned_question,
            primary_query=_require_string(payload, "primary_query"),
            additional_queries=additional_queries,
            keywords=tuple(payload.get("keywords", ())),
            hyde_document=hyde_document,
            strategy="llm",
        )


_SYSTEM_PROMPT = """你是论文知识库的检索查询规划器。
只输出 JSON 对象，不要输出 Markdown。改写不得改变用户意图，不得引入未出现的事实。
"""


def _build_user_prompt(question: str, config: QueryPlanningConfig) -> str:
    """构造受约束的查询规划请求。"""

    return f"""请为下列用户问题生成检索计划。

<question>{question}</question>

输出严格符合以下 JSON：
{{
  "primary_query": "用于主要检索的 query",
  "additional_queries": ["可选的互补 query，最多 {config.max_additional_queries} 条"],
  "keywords": ["英文或领域关键词"],
  "hyde_document": "仅在允许时给出的假想论文段落，否则为 null"
}}

multi_query_enabled={str(config.multi_query_enabled).lower()}
hyde_enabled={str(config.hyde_enabled).lower()}
"""


def _parse_payload(content: str) -> dict[str, Any]:
    """解析并初步校验 LLM 返回的 JSON 对象。"""

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("查询改写模型未返回合法 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("查询改写模型返回值必须是 JSON 对象")
    for name in ("additional_queries", "keywords"):
        value = payload.get(name, [])
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"查询改写字段 {name} 必须是字符串列表")
    hyde_document = payload.get("hyde_document")
    if hyde_document is not None and not isinstance(hyde_document, str):
        raise ValueError("查询改写字段 hyde_document 必须是字符串或 null")
    return payload


def _require_string(payload: dict[str, Any], name: str) -> str:
    """读取非空字符串字段。"""

    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"查询改写字段 {name} 必须是非空字符串")
    return value

