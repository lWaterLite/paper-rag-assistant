"""文本切分器。

Chunking 决定 RAG 的检索颗粒度。本模块提供可配置的 chunker，并尽量保留解析阶段
已经获得的 section、page、source metadata。
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from app.core.models import DocumentChunk, ParsedBlock, ParsedDocument
from app.ingest.chunking.metadata import ChunkMetadata, ChunkMetadataBuilder

ChunkingStrategy = str
TokenizerName = Literal["char_approx", "simple_regex"]


@dataclass(frozen=True)
class ChunkerConfig:
    """chunker 运行时配置。"""

    strategy: ChunkingStrategy = "section_aware"
    chunk_size: int = 600
    chunk_overlap: int = 100
    tokenizer: TokenizerName = "char_approx"

    def __post_init__(self) -> None:
        """校验 chunking 窗口，避免切分步长为 0 或负数。"""

        if not isinstance(self.strategy, str):
            raise ValueError("strategy 必须是字符串")
        normalized_strategy = self.strategy.strip()
        if not normalized_strategy:
            raise ValueError("strategy 不能为空")
        object.__setattr__(self, "strategy", normalized_strategy)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap 必须大于等于 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")


@dataclass(frozen=True)
class TextWindow:
    """一次文本窗口切分结果。"""

    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class SectionGroup:
    """按 section 聚合后的文本块。"""

    section: str | None
    text: str
    page_start: int | None
    page_end: int | None
    char_start: int | None
    char_end: int | None
    block_count: int


def _build_chunk_id(version_id: str, chunk_index: int, chunk_text: str) -> str:
    """生成稳定 chunk_id。"""

    digest = hashlib.sha1(
        f"{version_id}:{chunk_index}:{chunk_text}".encode("utf-8")
    ).hexdigest()[:12]
    return f"chunk_{digest}"


def estimate_token_count(text: str, tokenizer: TokenizerName) -> int:
    """估算文本 token 数。

    `char_approx` 适合当前无额外依赖的学习阶段；`simple_regex` 会把英文词、数字、
    CJK 字符和标点拆成较细粒度的 token。后续接真实 embedding 模型时，可以替换为
    provider 对应 tokenizer。
    """

    if tokenizer == "simple_regex":
        return len(_regex_token_spans(text))
    return len(text)


def _regex_token_spans(text: str) -> list[tuple[int, int]]:
    """用轻量规则切出 token span。

    这是无第三方 tokenizer 时的工程 fallback，不追求和真实 embedding 模型完全一致。
    规则按优先级排列：先匹配应该保持为整体的特殊 token，再匹配普通词、数字、
    中文字符和剩余非空白符号。
    """

    pattern = re.compile(
        r"""
        (?:[A-Za-z]\.){2,}              # 英文缩写，例如 U.S.A.、e.g.
        |\d+(?:[./:-]\d+)+              # 小数、日期或版本号，例如 3.14、2024-06
        |[A-Za-z]+(?:[-_'][A-Za-z]+)*   # 英文单词或连字符词，例如 retrieval-augmented
        |\d+                            # 普通整数
        |[\u4e00-\u9fff]                # CJK 字符，当前阶段按单字估算
        |\S                             # 兜底：标点、符号等非空白字符
        """,
        re.VERBOSE,
    )
    return [(match.start(), match.end()) for match in pattern.finditer(text)]


def _split_text_windows_by_chars(
    text: str, chunk_size: int, chunk_overlap: int
) -> list[TextWindow]:
    """按字符窗口切分文本。"""

    if not text:
        return []

    windows: list[TextWindow] = []
    step = chunk_size - chunk_overlap
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        raw_window = text[start:end]
        stripped_text = raw_window.strip()
        leading_spaces = len(raw_window) - len(raw_window.lstrip())
        trailing_spaces = len(raw_window) - len(raw_window.rstrip())
        if stripped_text:
            windows.append(
                TextWindow(
                    text=stripped_text,
                    char_start=start + leading_spaces,
                    char_end=end - trailing_spaces,
                )
            )
        start += step

    return windows


class Chunker(ABC):
    """切分器抽象基类。"""

    def __init__(self, config: ChunkerConfig) -> None:
        self._config = config
        self._chunk_size = config.chunk_size
        self._chunk_overlap = config.chunk_overlap
        self._metadata_builder = ChunkMetadataBuilder()

    @property
    def config(self) -> ChunkerConfig:
        """返回当前 chunker 使用的运行时配置。"""

        return self._config

    @abstractmethod
    def split(self, document: ParsedDocument) -> list[DocumentChunk]:
        """把解析后的文档切成 chunks。"""

    def _build_chunk(
        self,
        *,
        document: ParsedDocument,
        chunk_index: int,
        text: str,
        section: str | None = None,
        page_start: int | None = None,
        page_end: int | None = None,
        char_start: int | None = None,
        char_end: int | None = None,
        extra_metadata: dict[str, int | str | bool | None] | None = None,
    ) -> DocumentChunk:
        """统一构造 DocumentChunk，保证 metadata 形状一致。"""

        metadata = self._metadata_builder.build(
            document_metadata=document.metadata,
            chunk_metadata=ChunkMetadata(
                chunker=type(self).__name__,
                chunking_strategy=self._config.strategy,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                tokenizer=self._config.tokenizer,
                char_start=char_start,
                char_end=char_end,
                section_title=section,
            ),
            extra_metadata=extra_metadata,
        )

        return DocumentChunk(
            chunk_id=_build_chunk_id(document.version_id, chunk_index, text),
            doc_id=document.doc_id,
            content_hash=document.content_hash,
            version_id=document.version_id,
            text=text,
            source_path=document.source_path,
            chunk_index=chunk_index,
            token_count=estimate_token_count(text, self._config.tokenizer),
            title=document.title,
            section=section,
            page_start=page_start,
            page_end=page_end,
            metadata=metadata,
        )


class CharacterChunker(Chunker):
    """基于字符窗口的 baseline chunker。"""

    def split(self, document: ParsedDocument) -> list[DocumentChunk]:
        """将整篇文档按字符窗口切成 chunks。"""

        windows = self._split_text_by_chars(document.text)
        return [
            self._build_chunk(
                document=document,
                chunk_index=index,
                text=window.text,
                char_start=window.char_start,
                char_end=window.char_end,
            )
            for index, window in enumerate(windows)
        ]

    def _split_text_by_chars(self, text: str) -> list[TextWindow]:
        """按字符窗口切分文本。"""

        return _split_text_windows_by_chars(text, self._chunk_size, self._chunk_overlap)


class FixedTokenChunker(Chunker):
    """基于轻量 token span 的 chunker。

    这里不直接引入外部 tokenizer，避免子模块 3 一开始被依赖配置打断。
    后续接真实 embedding provider 时，可以把 tokenizer 抽象替换掉。
    """

    def split(self, document: ParsedDocument) -> list[DocumentChunk]:
        """将整篇文档按 token 窗口切成 chunks。"""

        spans = _regex_token_spans(document.text)
        if not spans:
            return []

        chunks: list[DocumentChunk] = []
        step = self._chunk_size - self._chunk_overlap
        token_start = 0

        while token_start < len(spans):
            token_end = min(token_start + self._chunk_size, len(spans))
            char_start = spans[token_start][0]
            char_end = spans[token_end - 1][1]
            chunk_text = document.text[char_start:char_end].strip()
            if chunk_text:
                chunks.append(
                    self._build_chunk(
                        document=document,
                        chunk_index=len(chunks),
                        text=chunk_text,
                        char_start=char_start,
                        char_end=char_end,
                        extra_metadata={
                            "token_start": token_start,
                            "token_end": token_end,
                        },
                    )
                )
            if token_end == len(spans):
                break
            token_start += step

        return chunks


class SectionAwareChunker(Chunker):
    """优先保留 section/page metadata 的 chunker。"""

    def split(self, document: ParsedDocument) -> list[DocumentChunk]:
        """按 section 聚合 ParsedBlock，再在 section 内部切分。"""

        if not document.text:
            return []

        section_groups = self._build_section_groups(document)
        chunks: list[DocumentChunk] = []

        for group in section_groups:
            for window in self._split_text_by_chars(group.text):
                chunks.append(
                    self._build_chunk(
                        document=document,
                        chunk_index=len(chunks),
                        text=window.text,
                        section=group.section,
                        page_start=group.page_start,
                        page_end=group.page_end,
                        char_start=_add_optional(group.char_start, window.char_start),
                        char_end=_add_optional(group.char_start, window.char_end),
                        extra_metadata={
                            "section_block_count": group.block_count,
                        },
                    )
                )

        return chunks

    def _split_text_by_chars(self, text: str) -> list[TextWindow]:
        """在单个 section 内部按字符窗口二次切分。"""

        return _split_text_windows_by_chars(text, self._chunk_size, self._chunk_overlap)

    @staticmethod
    def _build_section_groups(document: ParsedDocument) -> list[SectionGroup]:
        """从 ParsedBlock 构造 section groups。

        如果解析器没有提供 blocks，则退回到 Markdown 标题切分，保证旧文档仍可处理。
        """

        if not document.blocks:
            return SectionAwareChunker._split_markdown_sections(document.text)

        groups: list[SectionGroup] = []
        current_section: str | None = None
        current_blocks: list[ParsedBlock] = []

        for block in document.blocks:
            if block.section and block.section != current_section and current_blocks:
                groups.append(_build_section_group(current_section, current_blocks))
                current_blocks = []
            current_section = block.section or current_section
            current_blocks.append(block)

        if current_blocks:
            groups.append(_build_section_group(current_section, current_blocks))

        return groups

    @staticmethod
    def _split_markdown_sections(text: str) -> list[SectionGroup]:
        """按 Markdown 标题拆分没有 ParsedBlock 的文本。

        当前 fallback 只识别以 `#` 开头的 Markdown 标题。
        """

        sections: list[SectionGroup] = []
        current_lines: list[str] = []
        current_title: str | None = None
        current_start = 0
        cursor = 0

        for line in text.splitlines(keepends=True):
            line_without_newline = line.rstrip("\r\n")
            if line_without_newline.startswith("#") and current_lines:
                section_text = "".join(current_lines).strip()
                sections.append(
                    SectionGroup(
                        section=current_title,
                        text=section_text,
                        page_start=None,
                        page_end=None,
                        char_start=current_start,
                        char_end=current_start + len(section_text),
                        block_count=0,
                    )
                )
                current_lines = []
                current_start = cursor

            if line_without_newline.startswith("#"):
                current_title = line_without_newline.lstrip("#").strip() or None

            current_lines.append(line)
            cursor += len(line)

        if current_lines:
            section_text = "".join(current_lines).strip()
            sections.append(
                SectionGroup(
                    section=current_title,
                    text=section_text,
                    page_start=None,
                    page_end=None,
                    char_start=current_start,
                    char_end=current_start + len(section_text),
                    block_count=0,
                )
            )

        return sections


def _build_section_group(
    section: str | None, blocks: list[ParsedBlock]
) -> SectionGroup:
    """把同一个 section 下的 blocks 聚合为文本组。"""

    text = "\n\n".join(block.text.strip() for block in blocks if block.text.strip())
    page_values = [
        page
        for block in blocks
        for page in (block.page_start, block.page_end)
        if page is not None
    ]
    char_starts = [block.char_start for block in blocks if block.char_start is not None]
    char_ends = [block.char_end for block in blocks if block.char_end is not None]

    return SectionGroup(
        section=section,
        text=text,
        page_start=min(page_values) if page_values else None,
        page_end=max(page_values) if page_values else None,
        char_start=min(char_starts) if char_starts else None,
        char_end=max(char_ends) if char_ends else None,
        block_count=len(blocks),
    )


def _add_optional(base: int | None, offset: int) -> int | None:
    """只在 base 存在时做偏移计算。"""

    if base is None:
        return None
    return base + offset
