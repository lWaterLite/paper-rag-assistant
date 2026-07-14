"""文档摄取领域模型。

这些模型描述文档从加载到解析阶段的稳定数据边界。下游 indexing 和 retrieval
可以读取它们，但不应把 ingest 的领域模型放入 core。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class RawDocument:
    """从来源加载得到的原始文档。"""

    doc_id: str
    source_path: str
    file_type: str
    content_hash: str
    version_id: str
    raw_text: str
    raw_bytes: bytes | None = None
    source_uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


BlockType = Literal[
    "title",
    "heading",
    "paragraph",
    "list",
    "table",
    "code",
    "reference",
    "caption",
    "unknown",
]


@dataclass(frozen=True)
class ParseIssue:
    """解析或清洗阶段发现的质量问题。"""

    code: str
    message: str
    severity: Literal["info", "warning", "error"] = "warning"
    page: int | None = None
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedBlock:
    """解析后的结构化文本块。"""

    block_id: str
    doc_id: str
    version_id: str
    text: str
    block_type: BlockType
    source_path: str
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedDocument:
    """解析和清洗后的结构化文档。"""

    doc_id: str
    content_hash: str
    version_id: str
    title: str
    text: str
    source_path: str
    blocks: list[ParsedBlock] = field(default_factory=list)
    parse_issues: list[ParseIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
