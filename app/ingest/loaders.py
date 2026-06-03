"""文档加载器。

当前只实现本地 Markdown/TXT 加载，让你先理解 loading 阶段的输入输出。
PDF 和 HTML 会在后续子模块继续补充。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.core.errors import AppError, ErrorCode
from app.core.models import RawDocument


class LocalTextLoader:
    """加载本地 Markdown 和 TXT 文档。"""

    supported_suffixes = {".md", ".txt"}

    def load_directory(self, source_dir: Path) -> list[RawDocument]:
        """加载目录下所有支持的文本文件。"""

        if not source_dir.exists():
            raise AppError(ErrorCode.DOCUMENT_LOAD_FAILED, f"文档目录不存在：{source_dir}")

        documents: list[RawDocument] = []
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in self.supported_suffixes:
                continue
            documents.append(self.load_file(path))
        return documents

    def load_file(self, path: Path) -> RawDocument:
        """加载单个文件。"""

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise AppError(ErrorCode.DOCUMENT_LOAD_FAILED, f"文件不是有效 UTF-8 文本：{path}") from exc

        doc_id = self._build_doc_id(path)
        content_hash = self._build_content_hash(text)
        version_id = self._build_version_id(doc_id, content_hash)

        return RawDocument(
            doc_id=doc_id,
            source_path=str(path),
            file_type=path.suffix.lower().lstrip("."),
            content_hash=content_hash,
            version_id=version_id,
            raw_text=text,
            metadata={
                "filename": path.name,
                "suffix": path.suffix.lower(),
                "content_hash": content_hash,
                "version_id": version_id,
            },
        )

    @staticmethod
    def _build_doc_id(path: Path) -> str:
        """根据路径生成稳定 doc_id。

        TODO 练习 2：
        当前 doc_id 只基于路径生成。如果同一路径下文件内容改变，doc_id 不会改变。
        请你思考：真实系统中 doc_id 应该只和路径有关，还是应该和内容 hash、版本号有关？
        """

        normalized_path = path.resolve().as_posix()
        digest = hashlib.sha1(normalized_path.encode("utf-8")).hexdigest()[:12]
        return f"doc_{digest}"

    @staticmethod
    def _build_content_hash(text: str) -> str:
        """根据原始文本生成内容指纹。"""

        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _build_version_id(doc_id: str, content_hash: str) -> str:
        """根据文档身份和内容指纹生成版本 ID。"""

        digest = hashlib.sha1(f"{doc_id}:{content_hash}".encode("utf-8")).hexdigest()[:12]
        return f"v_{digest}"
