"""文档解析器。

子模块 2 将 parsing 层升级为真实文档解析：Markdown、HTML、PDF 都会转换成统一的
ParsedDocument，并尽量保留 block、页码、章节和解析质量问题。
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from html import unescape
from html.parser import HTMLParser
from typing import Protocol

from app.core.errors import AppError, ErrorCode
from app.core.models import BlockType, ParseIssue, ParsedBlock, ParsedDocument, RawDocument
from app.ingest.cleaners import BasicTextCleaner, HtmlTextCleaner, PdfTextCleaner


class DocumentParser(Protocol):
    """文档解析器接口。"""

    supported_file_types: set[str]

    def parse(self, document: RawDocument) -> ParsedDocument:
        """把 RawDocument 解析成 ParsedDocument。"""


def build_block_id(document: RawDocument, block_index: int, text: str) -> str:
    """生成稳定 block_id。"""

    import hashlib

    digest = hashlib.sha1(f"{document.version_id}:{block_index}:{text}".encode("utf-8")).hexdigest()[:12]
    return f"block_{digest}"


class ParserRegistry:
    """根据 file_type 选择对应解析器。"""

    def __init__(self, parsers: list[DocumentParser]) -> None:
        self._parsers = parsers

    def parse(self, document: RawDocument) -> ParsedDocument:
        """解析文档。"""

        for parser in self._parsers:
            if document.file_type in parser.supported_file_types:
                return parser.parse(document)
        raise AppError(ErrorCode.DOCUMENT_PARSE_FAILED, f"没有可用解析器处理文档类型：{document.file_type}")


class PlainTextParser:
    """解析普通纯文本文件。"""

    supported_file_types = {"txt", "text"}

    def __init__(self, cleaner: BasicTextCleaner) -> None:
        self._cleaner = cleaner

    def parse(self, document: RawDocument) -> ParsedDocument:
        """解析普通文本。"""

        cleaned = self._cleaner.clean(document.raw_text)
        title = self._guess_title(cleaned.text, document.metadata.get("filename", document.doc_id))
        blocks = self._build_paragraph_blocks(document, cleaned.text)

        return ParsedDocument(
            doc_id=document.doc_id,
            content_hash=document.content_hash,
            version_id=document.version_id,
            title=title,
            text=cleaned.text,
            source_path=document.source_path,
            blocks=blocks,
            parse_issues=cleaned.issues,
            metadata={
                **document.metadata,
                **cleaned.metadata,
                "title": title,
                "parsed_at": datetime.now(UTC).isoformat(),
                "parser": type(self).__name__,
            },
        )

    @staticmethod
    def _guess_title(text: str, fallback: str) -> str:
        """从文档中猜测标题。"""

        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped.lstrip("#").strip() or fallback
        return fallback

    @staticmethod
    def _clean_text(text: str) -> tuple[str, dict[str, int | str | bool]]:
        """兼容子模块 1 测试的清洗入口。"""

        cleaned = BasicTextCleaner().clean(text)
        return cleaned.text, cleaned.metadata

    @staticmethod
    def _build_paragraph_blocks(document: RawDocument, text: str) -> list[ParsedBlock]:
        """按空行切成段落块。"""

        blocks: list[ParsedBlock] = []
        cursor = 0
        for block_index, paragraph in enumerate(_split_paragraphs(text)):
            start = text.find(paragraph, cursor)
            end = start + len(paragraph)
            cursor = end
            blocks.append(
                ParsedBlock(
                    block_id=build_block_id(document, block_index, paragraph),
                    doc_id=document.doc_id,
                    version_id=document.version_id,
                    text=paragraph,
                    block_type="paragraph",
                    source_path=document.source_path,
                    char_start=start,
                    char_end=end,
                )
            )
        return blocks


class MarkdownParser(PlainTextParser):
    """解析 Markdown 文档。"""

    supported_file_types = {"markdown", "md"}

    def parse(self, document: RawDocument) -> ParsedDocument:
        """解析 Markdown，并提取 frontmatter、标题和 section。"""

        frontmatter, body = self._extract_frontmatter(document.raw_text)
        cleaned = self._cleaner.clean(body)
        title = str(frontmatter.get("title") or self._guess_title(cleaned.text, document.metadata.get("filename", document.doc_id)))
        blocks = self._build_markdown_blocks(document, cleaned.text)

        return ParsedDocument(
            doc_id=document.doc_id,
            content_hash=document.content_hash,
            version_id=document.version_id,
            title=title,
            text=cleaned.text,
            source_path=document.source_path,
            blocks=blocks,
            parse_issues=cleaned.issues,
            metadata={
                **document.metadata,
                **cleaned.metadata,
                **{f"frontmatter_{key}": value for key, value in frontmatter.items()},
                "title": title,
                "section_count": len({block.section for block in blocks if block.section}),
                "parsed_at": datetime.now(UTC).isoformat(),
                "parser": type(self).__name__,
            },
        )

    @staticmethod
    def _extract_frontmatter(text: str) -> tuple[dict[str, str], str]:
        """提取简单 YAML frontmatter。

        为避免在练习阶段直接强依赖 PyYAML，这里先支持常见的 key: value 形式。
        """

        if not text.startswith("---\n"):
            return {}, text

        end = text.find("\n---", 4)
        if end == -1:
            return {}, text

        raw_frontmatter = text[4:end]
        body = text[end + len("\n---") :].lstrip("\n")
        metadata: dict[str, str] = {}
        for line in raw_frontmatter.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip('"').strip("'")
        return metadata, body

    @staticmethod
    def _build_markdown_blocks(document: RawDocument, text: str) -> list[ParsedBlock]:
        """把 Markdown 文本切成带 section 的 block。"""

        blocks: list[ParsedBlock] = []
        current_section: str | None = None
        cursor = 0

        for block_index, paragraph in enumerate(_split_paragraphs(text)):
            block_type = _classify_markdown_block(paragraph)
            if block_type == "heading":
                current_section = paragraph.lstrip("#").strip()

            start = text.find(paragraph, cursor)
            end = start + len(paragraph)
            cursor = end
            blocks.append(
                ParsedBlock(
                    block_id=build_block_id(document, block_index, paragraph),
                    doc_id=document.doc_id,
                    version_id=document.version_id,
                    text=paragraph,
                    block_type=block_type,
                    source_path=document.source_path,
                    section=current_section,
                    char_start=start,
                    char_end=end,
                )
            )
        return blocks


class HtmlDocumentParser:
    """解析本地 HTML 文件。"""

    supported_file_types = {"html"}

    def __init__(self, cleaner: HtmlTextCleaner) -> None:
        self._cleaner = cleaner

    def parse(self, document: RawDocument) -> ParsedDocument:
        """解析 HTML 正文和 metadata。"""

        extractor = _ReadableHtmlExtractor()
        extractor.feed(document.raw_text)
        extracted_text = extractor.get_text()
        cleaned = self._cleaner.clean(extracted_text)
        title = extractor.title or self._guess_title(cleaned.text, document.metadata.get("filename", document.doc_id))
        blocks = self._build_html_blocks(document, cleaned.text)

        return ParsedDocument(
            doc_id=document.doc_id,
            content_hash=document.content_hash,
            version_id=document.version_id,
            title=title,
            text=cleaned.text,
            source_path=document.source_path,
            blocks=blocks,
            parse_issues=cleaned.issues,
            metadata={
                **document.metadata,
                **cleaned.metadata,
                "title": title,
                "canonical_url": extractor.canonical_url,
                "description": extractor.description,
                "parsed_at": datetime.now(UTC).isoformat(),
                "parser": type(self).__name__,
            },
        )

    @staticmethod
    def _guess_title(text: str, fallback: str) -> str:
        for line in text.splitlines():
            if line.strip():
                return line.strip()
        return fallback

    @staticmethod
    def _build_html_blocks(document: RawDocument, text: str) -> list[ParsedBlock]:
        blocks: list[ParsedBlock] = []
        cursor = 0
        current_section: str | None = None
        for block_index, paragraph in enumerate(_split_paragraphs(text)):
            block_type: BlockType = "heading" if len(paragraph) <= 120 and not paragraph.endswith((".", "。")) else "paragraph"
            if block_type == "heading":
                current_section = paragraph
            start = text.find(paragraph, cursor)
            end = start + len(paragraph)
            cursor = end
            blocks.append(
                ParsedBlock(
                    block_id=build_block_id(document, block_index, paragraph),
                    doc_id=document.doc_id,
                    version_id=document.version_id,
                    text=paragraph,
                    block_type=block_type,
                    source_path=document.source_path,
                    section=current_section,
                    char_start=start,
                    char_end=end,
                )
            )
        return blocks


class PdfDocumentParser:
    """解析真实 PDF 文件。"""

    supported_file_types = {"pdf"}

    def __init__(self, cleaner: PdfTextCleaner) -> None:
        self._cleaner = cleaner

    def parse(self, document: RawDocument) -> ParsedDocument:
        """解析 PDF。

        当前项目统一使用 PyMuPDF 作为 PDF 解析基础。
        它能提供页面、文本块和坐标等信息，后续更适合继续扩展 layout-aware 解析。
        """

        if not document.raw_bytes:
            raise AppError(ErrorCode.DOCUMENT_PARSE_FAILED, f"PDF 文档缺少原始字节：{document.source_path}")

        pages, pdf_metadata = self._extract_pages(document.raw_bytes, document.source_path)
        if not any(text.strip() for _, text in pages):
            issue = ParseIssue(
                code="pdf_no_extractable_text",
                message="PDF 没有可提取文本，可能是扫描版或受保护文档",
                severity="error",
            )
            return self._build_empty_pdf_document(document, pdf_metadata, [issue])

        cleaned = self._cleaner.clean_pages(pages)
        title = str(pdf_metadata.get("title") or document.metadata.get("filename", document.doc_id))
        blocks = self._build_pdf_blocks(document, pages)

        return ParsedDocument(
            doc_id=document.doc_id,
            content_hash=document.content_hash,
            version_id=document.version_id,
            title=title,
            text=cleaned.text,
            source_path=document.source_path,
            blocks=blocks,
            parse_issues=cleaned.issues,
            metadata={
                **document.metadata,
                **cleaned.metadata,
                **{f"pdf_{key}": value for key, value in pdf_metadata.items() if value},
                "title": title,
                "parsed_at": datetime.now(UTC).isoformat(),
                "parser": type(self).__name__,
            },
        )

    def _extract_pages(self, raw_bytes: bytes, source_path: str) -> tuple[list[tuple[int, str]], dict[str, str]]:
        """使用 PyMuPDF 从 PDF 字节中提取每页文本。"""

        try:
            return self._extract_pages_with_pymupdf(raw_bytes)
        except ImportError as exc:
            raise AppError(
                ErrorCode.DOCUMENT_PARSE_FAILED,
                "解析 PDF 需要安装 PyMuPDF。建议执行：uv add pymupdf，或 pip install pymupdf",
            ) from exc
        except Exception as exc:
            raise AppError(ErrorCode.DOCUMENT_PARSE_FAILED, f"PyMuPDF 解析 PDF 失败：{source_path}") from exc

    @staticmethod
    def _extract_pages_with_pymupdf(raw_bytes: bytes) -> tuple[list[tuple[int, str]], dict[str, str]]:
        import fitz

        pages: list[tuple[int, str]] = []
        metadata: dict[str, str] = {}
        with fitz.open(stream=raw_bytes, filetype="pdf") as pdf:
            metadata = {key: str(value) for key, value in pdf.metadata.items() if value}
            for index, page in enumerate(pdf, start=1):
                pages.append((index, page.get_text("text")))
        return pages, metadata

    def _build_pdf_blocks(self, document: RawDocument, pages: list[tuple[int, str]]) -> list[ParsedBlock]:
        """按页和段落生成 PDF blocks。"""

        blocks: list[ParsedBlock] = []
        block_index = 0
        current_section: str | None = None
        for page_number, page_text in pages:
            page_cleaned = self._cleaner.clean(self._cleaner.merge_pdf_line_breaks(page_text))
            for paragraph in _split_paragraphs(page_cleaned.text):
                block_type: BlockType = "heading" if _looks_like_pdf_heading(paragraph) else "paragraph"
                if block_type == "heading":
                    current_section = paragraph
                blocks.append(
                    ParsedBlock(
                        block_id=build_block_id(document, block_index, paragraph),
                        doc_id=document.doc_id,
                        version_id=document.version_id,
                        text=paragraph,
                        block_type=block_type,
                        source_path=document.source_path,
                        page_start=page_number,
                        page_end=page_number,
                        section=current_section,
                    )
                )
                block_index += 1
        return blocks

    @staticmethod
    def _build_empty_pdf_document(
        document: RawDocument,
        pdf_metadata: dict[str, str],
        issues: list[ParseIssue],
    ) -> ParsedDocument:
        """构造无法提取文本的 PDF 解析结果。"""

        title = str(pdf_metadata.get("title") or document.metadata.get("filename", document.doc_id))
        return ParsedDocument(
            doc_id=document.doc_id,
            content_hash=document.content_hash,
            version_id=document.version_id,
            title=title,
            text="",
            source_path=document.source_path,
            blocks=[],
            parse_issues=issues,
            metadata={
                **document.metadata,
                **{f"pdf_{key}": value for key, value in pdf_metadata.items() if value},
                "title": title,
                "parsed_at": datetime.now(UTC).isoformat(),
                "parser": "PdfDocumentParser",
            },
        )


class _ReadableHtmlExtractor(HTMLParser):
    """从 HTML 中提取正文和基础 metadata。"""

    ignored_tags = {"script", "style", "noscript", "svg", "nav", "footer", "header"}
    block_tags = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "td", "th"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: str | None = None
        self.description: str | None = None
        self.canonical_url: str | None = None
        self._tag_stack: list[str] = []
        self._current_block: list[str] = []
        self._blocks: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key.lower(): value for key, value in attrs}
        self._tag_stack.append(tag)
        if tag == "title":
            self._in_title = True
        if tag == "meta":
            name = (attrs_dict.get("name") or attrs_dict.get("property") or "").lower()
            content = attrs_dict.get("content")
            if name in {"description", "og:description"} and content:
                self.description = content.strip()
        if tag == "link" and (attrs_dict.get("rel") or "").lower() == "canonical":
            self.canonical_url = attrs_dict.get("href")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in self.block_tags:
            self._flush_block()
        if self._tag_stack:
            self._tag_stack.pop()

    def handle_data(self, data: str) -> None:
        if self._is_ignored():
            return
        text = unescape(data).strip()
        if not text:
            return
        if self._in_title:
            self.title = text if self.title is None else f"{self.title} {text}"
            return
        self._current_block.append(text)

    def get_text(self) -> str:
        self._flush_block()
        return "\n\n".join(block for block in self._blocks if block)

    def _is_ignored(self) -> bool:
        return any(tag in self.ignored_tags for tag in self._tag_stack)

    def _flush_block(self) -> None:
        if not self._current_block:
            return
        text = " ".join(part.strip() for part in self._current_block if part.strip())
        if text:
            self._blocks.append(text)
        self._current_block = []


def _split_paragraphs(text: str) -> list[str]:
    """按空行切分段落。"""

    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def _classify_markdown_block(text: str) -> BlockType:
    """粗略识别 Markdown block 类型。"""

    stripped = text.strip()
    if stripped.startswith("#"):
        return "heading"
    if stripped.startswith("```"):
        return "code"
    if stripped.startswith(("- ", "* ")) or re.match(r"^\d+\.\s+", stripped):
        return "list"
    if "|" in stripped and "\n" in stripped:
        return "table"
    return "paragraph"


def _looks_like_pdf_heading(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) > 120:
        return False
    return bool(re.match(r"^(\d+(\.\d+)*)?\s*(Abstract|Introduction|Related Work|References|Conclusion)\b", stripped, re.I))
