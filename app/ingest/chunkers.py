"""文本切分器。

chunking 是 RAG 中非常关键的一步。这里先用字符切分建立直觉。
"""

from __future__ import annotations

from app.core.config import Settings
from app.core.errors import AppError, ErrorCode
from app.core.models import DocumentChunk, ParsedDocument, new_id


class CharacterChunker:
    """基于字符长度的简单 chunker。"""

    def __init__(self, settings: Settings) -> None:
        if settings.chunk_size <= 0:
            raise AppError(ErrorCode.CHUNK_FAILED, "chunk_size 必须大于 0")
        if settings.chunk_overlap >= settings.chunk_size:
            raise AppError(ErrorCode.CHUNK_FAILED, "chunk_overlap 必须小于 chunk_size")
        self._settings = settings

    def split(self, document: ParsedDocument) -> list[DocumentChunk]:
        """将文档切分为多个 DocumentChunk。"""

        text = document.text
        if not text:
            return []

        chunks: list[DocumentChunk] = []
        start = 0
        chunk_index = 0
        step = self._settings.chunk_size - self._settings.chunk_overlap

        while start < len(text):
            end = min(start + self._settings.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        chunk_id=new_id("chunk"),
                        doc_id=document.doc_id,
                        text=chunk_text,
                        source_path=document.source_path,
                        chunk_index=chunk_index,
                        token_count=len(chunk_text),
                        title=document.title,
                        metadata={
                            **document.metadata,
                            "char_start": start,
                            "char_end": end,
                        },
                    )
                )
                chunk_index += 1
            start += step

        return chunks

    # TODO 练习 4：
    # 当前切分完全不理解语义边界，可能把一句话或一个 Markdown 小节切断。
    # 请你尝试实现一个 SectionAwareChunker：
    # 1. 优先按 Markdown 标题切分。
    # 2. 再把过长小节切成多个 chunk。
    # 3. 在 metadata 中记录 section 标题。

