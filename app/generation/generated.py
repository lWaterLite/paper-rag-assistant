"""LLM 输出到生成领域模型之间的受限载荷。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GeneratedAnswerPayload:
    """模型返回的结构化回答，尚未完成 citation 校验。"""

    answer: str
    citation_ids: tuple[str, ...]
    abstained: bool
    abstention_reason: str | None

    @classmethod
    def from_json(cls, content: str) -> "GeneratedAnswerPayload":
        """解析模型 JSON，并校验字段类型。"""

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("回答模型未返回合法 JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("回答模型返回值必须是 JSON 对象")

        answer = _require_non_blank_string(payload, "answer")
        citation_ids = payload.get("citation_ids")
        if not isinstance(citation_ids, list) or not all(
            isinstance(item, str) for item in citation_ids
        ):
            raise ValueError("回答字段 citation_ids 必须是字符串列表")
        abstained = payload.get("abstained")
        if not isinstance(abstained, bool):
            raise ValueError("回答字段 abstained 必须是布尔值")
        reason = payload.get("abstention_reason")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("回答字段 abstention_reason 必须是字符串或 null")
        return cls(
            answer=answer,
            citation_ids=tuple(citation_ids),
            abstained=abstained,
            abstention_reason=reason.strip() if isinstance(reason, str) else None,
        )


def _require_non_blank_string(payload: dict[str, Any], name: str) -> str:
    """读取非空字符串字段。"""

    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"回答字段 {name} 必须是非空字符串")
    return value.strip()

