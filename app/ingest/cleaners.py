"""文档清洗器。

清洗的目标不是把文本压到最短，而是在保留引用信息的前提下降低噪声。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
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

        return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"))

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


class PdfTextCleaner(BasicTextCleaner):
    """PDF 文本清洗器。

    这里实现的是保守规则：只修复非常常见、风险相对低的问题。
    """

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
            fixed_line_breaks = self._merge_pdf_line_breaks(without_headers)
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
    def _merge_pdf_line_breaks(text: str) -> str:
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

            if current and not _looks_like_sentence_boundary(current) and not _looks_like_heading(line):
                current = f"{current} {line}"
            else:
                if current:
                    paragraphs.append(current.strip())
                current = line

        if current:
            paragraphs.append(current.strip())

        return "\n\n".join(paragraphs)

    @staticmethod
    def _detect_repeated_edge_lines(pages: list[tuple[int, str]]) -> set[str]:
        """检测多页重复出现的顶部/底部短文本。"""

        if len(pages) < 3:
            return set()

        counter: Counter[str] = Counter()
        for _, text in pages:
            non_empty = [line.strip() for line in text.splitlines() if line.strip()]
            edge_lines = [*non_empty[:2], *non_empty[-2:]]
            for line in edge_lines:
                if 3 <= len(line) <= 120:
                    counter[line] += 1

        threshold = max(2, len(pages) // 2)
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


# TODO 子模块2-练习2：
# 当前 PDF 页眉页脚检测只看页面顶部和底部各两行。
# 请你把它改造成可配置策略，例如 edge_line_count=3、min_repeat_ratio=0.6，
# 并让调用方可以根据不同论文版式调整清洗强度。
