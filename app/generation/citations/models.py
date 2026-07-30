"""回答 citation 校验的中间模型。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CitationValidationResult:
    """citation 校验成功后的规范化结果。"""

    citation_ids: tuple[str, ...]
    inline_citation_ids: tuple[str, ...]
