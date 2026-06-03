"""文档解析器。

子模块 1 中先用纯文本解析器理解 ParsedDocument 的形状。
"""

from __future__ import annotations

from app.core.models import ParsedDocument, RawDocument


class PlainTextParser:
    """把 RawDocument 转换为 ParsedDocument。"""

    def parse(self, document: RawDocument) -> ParsedDocument:
        """解析纯文本或 Markdown 文档。"""

        title = self._guess_title(document.raw_text, document.metadata.get("filename", document.doc_id))
        cleaned_text, cleaning_metadata = self._clean_text(document.raw_text)
        return ParsedDocument(
            doc_id=document.doc_id,
            content_hash=document.content_hash,
            version_id=document.version_id,
            title=title,
            text=cleaned_text,
            source_path=document.source_path,
            metadata={
                **document.metadata,
                **cleaning_metadata,
            },
        )

    @staticmethod
    def _guess_title(text: str, fallback: str) -> str:
        """从文档中猜测标题。"""

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            return stripped.lstrip("#").strip() or fallback
        return fallback

    @staticmethod
    def _clean_text(text: str) -> tuple[str, dict[str, int]]:
        """基础清洗。

        TODO 练习 3：
        当前清洗只去掉首尾空白。
        请你补充至少两条清洗规则，例如：
        1. 合并连续空行。
        2. 去除行尾多余空格。
        3. 保留 Markdown 标题但规范化空白。
        """

        lines = text.splitlines()
        cleaned_lines: list[str] = []
        previous_blank = False
        for line in lines:
            cleaned_line = line.rstrip()
            title_candidate = cleaned_line.strip()

            if title_candidate.startswith("#"):
                parts = title_candidate.split(maxsplit=1)
                cleaned_line = f"{parts[0]} {parts[1].strip()}" if len(parts) == 2 else parts[0]

            is_blank = cleaned_line.strip() == ""
            if is_blank and previous_blank:
                continue

            cleaned_lines.append(cleaned_line)
            previous_blank = is_blank

        cleaned_text = "\n".join(cleaned_lines).strip()

        return cleaned_text, {
            "raw_text_length": len(text),
            "cleaned_text_length": len(cleaned_text),
        }
