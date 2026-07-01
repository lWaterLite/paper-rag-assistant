"""chunk metadata 契约与构造器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from app.core.metadata import BaseMetadata


@dataclass(frozen=True)
class ChunkMetadata(BaseMetadata):
    """DocumentChunk.metadata 中由 chunking 子系统维护的标准字段。"""

    chunker: str
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    tokenizer: str
    char_start: int | None = None
    char_end: int | None = None
    section_title: str | None = None
    token_start: int | None = None
    token_end: int | None = None
    section_block_count: int | None = None


class ChunkMetadataBuilder:
    """组合文档 metadata、标准 chunk metadata 和策略扩展 metadata。"""

    @staticmethod
    def build(
        *,
        document_metadata: Mapping[str, Any],
        chunk_metadata: ChunkMetadata,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """构造最终写入 DocumentChunk.metadata 的字典。

        合并顺序体现优先级：原始文档 metadata -> 标准 chunk metadata -> 策略扩展 metadata。
        """

        metadata = dict(document_metadata)
        metadata.update(chunk_metadata.to_dict())
        if extra_metadata:
            metadata.update({key: value for key, value in extra_metadata.items() if value is not None})
        return metadata
