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
        cleaned_text = self._clean_text(document.raw_text)
        return ParsedDocument(
            doc_id=document.doc_id,
            content_hash=document.content_hash,
            version_id=document.version_id,
            title=title,
            text=cleaned_text,
            source_path=document.source_path,
            metadata=document.metadata,
        )

    def _guess_title(self, text: str, fallback: str) -> str:
        """从文档中猜测标题。"""

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            return stripped.lstrip("#").strip() or fallback
        return fallback

    def _clean_text(self, text: str) -> str:
        """基础清洗。

        TODO 练习 3：
        当前清洗只去掉首尾空白。
        请你补充至少两条清洗规则，例如：
        1. 合并连续空行。
        2. 去除行尾多余空格。
        3. 保留 Markdown 标题但规范化空白。
        """

        return text.strip()
