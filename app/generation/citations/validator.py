"""回答文本与上下文来源映射之间的确定性校验。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.generation.configuration import CitationValidationConfig
from app.generation.generated import GeneratedAnswerPayload
from app.generation.citations.models import CitationValidationResult
from app.retrieval.context.packer import ContextCitation


class CitationValidationError(ValueError):
    """模型输出不满足 citation 契约。"""


_INLINE_CITATION_PATTERN = re.compile(r"\[([A-Za-z][A-Za-z0-9_-]*)\]")


@dataclass(frozen=True, slots=True)
class CitationValidator:
    """验证 citation 格式、允许来源和拒答语义。"""

    config: CitationValidationConfig

    def validate(
        self,
        payload: GeneratedAnswerPayload,
        allowed_citations: list[ContextCitation],
    ) -> CitationValidationResult:
        """验证模型声明的 citation id 均来自本次上下文。"""

        allowed_ids = {citation.citation_id for citation in allowed_citations}
        inline_ids = _extract_distinct_inline_ids(payload.answer)
        payload_ids = _normalize_ids(payload.citation_ids)

        if payload.abstained:
            self._validate_abstention(payload, inline_ids, payload_ids)
            return CitationValidationResult((), ())

        if not allowed_ids:
            raise CitationValidationError("没有可用证据时，非拒答回答不合法")
        if self.config.require_citations_when_evidence and not payload_ids:
            raise CitationValidationError("非拒答回答必须声明至少一个 citation id")
        if self.config.require_inline_ids_match_payload and set(inline_ids) != set(payload_ids):
            raise CitationValidationError("回答正文中的 citation id 必须与 citation_ids 完全一致")

        unknown_ids = (set(inline_ids) | set(payload_ids)) - allowed_ids
        if unknown_ids:
            rendered_ids = ", ".join(sorted(unknown_ids))
            raise CitationValidationError(f"回答引用了当前上下文不存在的 citation id：{rendered_ids}")
        return CitationValidationResult(payload_ids, inline_ids)

    def _validate_abstention(
        self,
        payload: GeneratedAnswerPayload,
        inline_ids: tuple[str, ...],
        payload_ids: tuple[str, ...],
    ) -> None:
        """拒答不能伪装成已获证据支持的结论。"""

        if inline_ids or payload_ids:
            raise CitationValidationError("拒答回答不应声明 citation id")
        if self.config.require_abstention_reason and not payload.abstention_reason:
            raise CitationValidationError("拒答回答必须说明资料不足原因")


def _extract_distinct_inline_ids(answer: str) -> tuple[str, ...]:
    """按出现顺序提取正文中的 citation id。"""

    return _normalize_ids(_INLINE_CITATION_PATTERN.findall(answer))


def _normalize_ids(ids: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """清理 citation id，拒绝空值并保持首次出现顺序。"""

    normalized: list[str] = []
    seen: set[str] = set()
    for citation_id in ids:
        cleaned = citation_id.strip()
        if not cleaned:
            raise CitationValidationError("citation id 不能为空")
        if cleaned not in seen:
            seen.add(cleaned)
            normalized.append(cleaned)
    return tuple(normalized)

