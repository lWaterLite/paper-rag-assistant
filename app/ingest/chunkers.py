"""文本切分器。

chunking 是 RAG 中非常关键的一步。这里先用字符切分建立直觉。
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

from app.core.config import Settings
from app.core.models import DocumentChunk, ParsedDocument


def _build_chunk_id(version_id: str, chunk_index: int, chunk_text: str) -> str:
    """生成稳定 chunk_id。

    chunk_id 绑定 version_id、chunk_index 和 chunk_text。
    文档内容变化时 version_id 会变化，因此 chunk_id 也会变化，方便后续做 embedding cache。
    """

    digest = hashlib.sha1(f"{version_id}:{chunk_index}:{chunk_text}".encode("utf-8")).hexdigest()[:12]
    return f"chunk_{digest}"


class Chunker(ABC):
    """切分器抽象基类。"""

    def __init__(self, settings: Settings) -> None:
        self._chunk_size = settings.chunk_size
        self._chunk_overlap = settings.chunk_overlap

    @abstractmethod
    def split(self, document: ParsedDocument) -> list[DocumentChunk]:
        pass


class CharacterChunker(Chunker):
    """基于字符长度的简单 chunker。"""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)

    def split(self, document: ParsedDocument) -> list[DocumentChunk]:
        """将文档切分为多个 DocumentChunk。"""

        text = document.text
        if not text:
            return []

        chunks: list[DocumentChunk] = []
        start = 0
        chunk_index = 0
        step = self._chunk_size - self._chunk_overlap

        while start < len(text):
            end = min(start + self._chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        chunk_id=_build_chunk_id(document.version_id, chunk_index, chunk_text),
                        doc_id=document.doc_id,
                        content_hash=document.content_hash,
                        version_id=document.version_id,
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


class SectionAwareChunker(Chunker):
    """优先按 Markdown 标题切分的 chunker。

    如果某个小节超过 chunk_size，再对这个小节内部进行字符级二次切分。
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)

    def split(self, document: ParsedDocument) -> list[DocumentChunk]:
        text = document.text
        if not text:
            return []

        sections = self._split_sections(text)

        chunk_index = 0
        chunks: list[DocumentChunk] = []

        for section_text, section_title, section_start in sections:
            for chunk_text, relative_start, relative_end in self._split_section_text(section_text):
                chunks.append(
                    DocumentChunk(
                        chunk_id=_build_chunk_id(document.version_id, chunk_index, chunk_text),
                        doc_id=document.doc_id,
                        content_hash=document.content_hash,
                        version_id=document.version_id,
                        text=chunk_text,
                        source_path=document.source_path,
                        chunk_index=chunk_index,
                        token_count=len(chunk_text),
                        title=document.title,
                        section=section_title,
                        metadata={
                            **document.metadata,
                            "char_start": section_start + relative_start,
                            "char_end": section_start + relative_end,
                            "section_title": section_title,
                        },
                    )
                )
                chunk_index += 1

        return chunks

    @staticmethod
    def _split_sections(text: str) -> list[tuple[str, str, int]]:
        """按 Markdown 标题拆分文本。

        返回值为：
        1. 小节文本。
        2. 小节标题。
        3. 小节在原文中的起始字符位置。
        """

        sections: list[tuple[str, str, int]] = []
        current_lines: list[str] = []
        current_title = "untitled"
        current_start = 0
        cursor = 0

        for line in text.splitlines(keepends=True):
            line_without_newline = line.rstrip("\r\n")
            if line_without_newline.startswith("#") and current_lines:
                sections.append(("".join(current_lines).strip(), current_title, current_start))
                current_lines = []
                current_start = cursor

            if line_without_newline.startswith("#"):
                current_title = line_without_newline.lstrip("#").strip() or "untitled"

            current_lines.append(line)
            cursor += len(line)

        if current_lines:
            sections.append(("".join(current_lines).strip(), current_title, current_start))

        return sections

    def _split_section_text(self, section_text: str) -> list[tuple[str, int, int]]:
        """对单个小节进行二次切分。"""

        if len(section_text) <= self._chunk_size:
            return [(section_text, 0, len(section_text))]

        chunks: list[tuple[str, int, int]] = []
        step = self._chunk_size - self._chunk_overlap
        start = 0

        while start < len(section_text):
            end = min(start + self._chunk_size, len(section_text))
            chunk_text = section_text[start:end].strip()
            leading_spaces = len(section_text[start:end]) - len(section_text[start:end].lstrip())
            trailing_spaces = len(section_text[start:end]) - len(section_text[start:end].rstrip())
            if chunk_text:
                chunks.append((chunk_text, start + leading_spaces, end - trailing_spaces))
            start += step

        return chunks

