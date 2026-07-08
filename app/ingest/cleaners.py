"""文档清洗器。

清洗的目标不是把文本压到最短，而是在保留引用信息的前提下降低噪声。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from math import ceil
from typing import Protocol

from app.core.models import ParseIssue


@dataclass(frozen=True)
class CleanedText:
    """清洗后的文本与清洗报告。"""

    text: str
    issues: list[ParseIssue] = field(default_factory=list)
    metadata: dict[str, int | str | bool] = field(default_factory=dict)


class DocumentCleaner(Protocol):
    """文档清洗器接口。"""

    def clean(self, text: str) -> CleanedText:
        """清洗文本。"""


class BasicTextCleaner:
    """通用文本清洗器。"""

    def clean(self, text: str) -> CleanedText:
        """清洗普通文本和 Markdown。"""

        normalized = self._normalize_line_endings(text)
        normalized = self._normalize_markdown_headings(normalized)
        normalized = self._collapse_blank_lines(normalized)
        normalized = normalized.strip()

        return CleanedText(
            text=normalized,
            metadata={
                "raw_text_length": len(text),
                "cleaned_text_length": len(normalized),
                "cleaner": type(self).__name__,
            },
        )

    @staticmethod
    def _normalize_line_endings(text: str) -> str:
        """统一换行并去除行尾空白。"""

        return "\n".join(
            line.rstrip()
            for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        )

    @staticmethod
    def _collapse_blank_lines(text: str) -> str:
        """把连续三个及以上空行压缩为一个空行。"""

        return re.sub(r"\n{3,}", "\n\n", text)

    @staticmethod
    def _normalize_markdown_headings(text: str) -> str:
        """规范 Markdown 标题中的多余空白。"""

        cleaned_lines: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                parts = stripped.split(maxsplit=1)
                line = f"{parts[0]} {parts[1].strip()}" if len(parts) == 2 else parts[0]
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)


@dataclass(frozen=True)
class PdfTextCleanerConfig:
    """PDF 页眉页脚检测配置。"""

    edge_line_count: int = 2
    min_repeat_ratio: float = 0.6
    min_line_length: int = 3
    max_line_length: int = 120

    def __post_init__(self) -> None:
        """校验 PDF 清洗配置，避免无意义的阈值进入清洗逻辑。"""

        if self.edge_line_count <= 0:
            raise ValueError("edge_line_count 必须大于 0")
        if not 0 < self.min_repeat_ratio <= 1:
            raise ValueError("min_repeat_ratio 必须在 (0, 1] 范围内")
        if self.min_line_length <= 0:
            raise ValueError("min_line_length 必须大于 0")
        if self.max_line_length < self.min_line_length:
            raise ValueError("max_line_length 必须大于等于 min_line_length")


class PdfTextCleaner(BasicTextCleaner):
    """PDF 文本清洗器。

    这里实现的是保守规则：只修复非常常见、风险相对低的问题。
    """

    def __init__(self, config: PdfTextCleanerConfig):
        self._config = config

    def clean_pages(self, pages: list[tuple[int, str]]) -> CleanedText:
        """清洗按页提取出的 PDF 文本。"""

        issues: list[ParseIssue] = []
        header_footer_candidates = self._detect_repeated_edge_lines(pages)
        cleaned_page_texts: list[str] = []

        for page_number, page_text in pages:
            without_headers = self._remove_repeated_edge_lines(
                page_text,
                header_footer_candidates,
                page_number,
                issues,
            )
            fixed_line_breaks = self.merge_pdf_line_breaks(without_headers)
            cleaned_page_texts.append(fixed_line_breaks.strip())

        merged = "\n\n".join(text for text in cleaned_page_texts if text)
        cleaned = super().clean(merged)

        return CleanedText(
            text=cleaned.text,
            issues=[*issues, *cleaned.issues],
            metadata={
                **cleaned.metadata,
                "page_count": len(pages),
                "removed_repeated_edge_lines": len(header_footer_candidates),
            },
        )

    @staticmethod
    def merge_pdf_line_breaks(text: str) -> str:
        """合并 PDF 中常见的段内错误换行。"""

        lines = [line.strip() for line in text.splitlines()]
        paragraphs: list[str] = []
        current = ""

        for line in lines:
            if not line:
                if current:
                    paragraphs.append(current.strip())
                    current = ""
                continue

            if current.endswith("-") and line and line[0].islower():
                current = current[:-1] + line
                continue

            if (
                current
                and not _looks_like_sentence_boundary(current)
                and not _looks_like_heading(line)
            ):
                current = f"{current} {line}"
            else:
                if current:
                    paragraphs.append(current.strip())
                current = line

        if current:
            paragraphs.append(current.strip())

        return "\n\n".join(paragraphs)

    def _detect_repeated_edge_lines(self, pages: list[tuple[int, str]]) -> set[str]:
        """检测多页重复出现的顶部/底部短文本。"""

        min_line_length = self._config.min_line_length
        max_line_length = self._config.max_line_length
        edge_line_count = self._config.edge_line_count
        min_repeat_ratio = self._config.min_repeat_ratio

        if len(pages) < 3:
            return set()

        counter: Counter[str] = Counter()
        for _, text in pages:
            non_empty = [line.strip() for line in text.splitlines() if line.strip()]
            edge_lines = [*non_empty[:edge_line_count], *non_empty[-edge_line_count:]]
            for line in dict.fromkeys(edge_lines):
                if min_line_length <= len(line) <= max_line_length:
                    counter[line] += 1

        threshold = max(2, ceil(len(pages) * min_repeat_ratio))
        return {line for line, count in counter.items() if count >= threshold}

    @staticmethod
    def _remove_repeated_edge_lines(
        text: str,
        repeated_lines: set[str],
        page_number: int,
        issues: list[ParseIssue],
    ) -> str:
        """移除检测出的重复页眉页脚。"""

        if not repeated_lines:
            return text

        kept_lines: list[str] = []
        removed = 0
        for line in text.splitlines():
            if line.strip() in repeated_lines:
                removed += 1
                continue
            kept_lines.append(line)

        if removed:
            issues.append(
                ParseIssue(
                    code="pdf_repeated_edge_lines_removed",
                    message=f"第 {page_number} 页移除了 {removed} 行疑似重复页眉页脚",
                    severity="info",
                    page=page_number,
                )
            )

        return "\n".join(kept_lines)


class HtmlTextCleaner(BasicTextCleaner):
    """HTML 正文文本清洗器。"""

    def clean(self, text: str) -> CleanedText:
        cleaned = super().clean(text)
        return CleanedText(
            text=cleaned.text,
            issues=cleaned.issues,
            metadata={
                **cleaned.metadata,
                "html_whitespace_normalized": True,
            },
        )


def _looks_like_sentence_boundary(text: str) -> bool:
    """判断一行是否像自然段结束。"""

    return text.endswith((".", "?", "!", "。", "？", "！", ":", "："))


def _looks_like_heading(text: str) -> bool:
    """粗略判断 PDF 抽取行是否像标题。"""

    stripped = text.strip()
    if not stripped:
        return False
    if re.match(r"^\d+(\.\d+)*\s+\S+", stripped):
        return True
    return stripped.isupper() and len(stripped) <= 120
