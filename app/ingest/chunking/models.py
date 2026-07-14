"""文档切分阶段的领域模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentChunk:
    """文档切分后的检索基本单位。"""

    chunk_id: str
    doc_id: str
    content_hash: str
    version_id: str
    text: str
    source_path: str
    chunk_index: int
    token_count: int
    title: str | None = None
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
