"""文档加载器。

子模块 2 开始，loading 层需要面对真实文件，而不是只读 demo 文本。
本模块负责把本地 PDF、Markdown、HTML、TXT 文件读入系统，并生成稳定文档身份。
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from app.core.errors import AppError, ErrorCode
from app.core.models import RawDocument


@dataclass(frozen=True)
class DocumentSource:
    """本地文档来源。

    这里先只支持本地文件路径。后续接 URL、对象存储或数据库时，可以扩展 source_uri。
    """

    path: Path
    source_uri: str | None = None


class DocumentLoader(Protocol):
    """文档加载器接口。"""

    supported_suffixes: set[str]

    def load_file(self, path: Path) -> RawDocument:
        """加载单个文件。"""

    def load_directory(self, source_dir: Path) -> list[RawDocument]:
        """加载目录中的文件。"""


class DocumentIdentityBuilder:
    """文档身份生成器。

    doc_id 代表“这份文档是谁”，content_hash 代表“这次读到的内容是什么”，
    version_id 代表“这份文档的这个内容版本”。这三个概念后续会服务增量索引。
    """

    @staticmethod
    def build_doc_id(path: Path, source_uri: str | None = None) -> str:
        """根据规范化来源生成稳定 doc_id。"""

        source_identity = source_uri or path.resolve().as_posix()
        digest = hashlib.sha1(source_identity.encode("utf-8")).hexdigest()[:12]
        return f"doc_{digest}"

    @staticmethod
    def build_content_hash(content: bytes) -> str:
        """根据原始字节生成内容指纹。"""

        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def build_version_id(doc_id: str, content_hash: str) -> str:
        """根据文档身份和内容指纹生成版本 ID。"""

        digest = hashlib.sha1(f"{doc_id}:{content_hash}".encode("utf-8")).hexdigest()[:12]
        return f"v_{digest}"


class LocalDocumentLoader:
    """加载本地论文和文档文件。

    PDF 是二进制格式，loading 阶段只读取字节，不在这里做文本提取；
    Markdown、HTML、TXT 会同时保存 raw_text，方便 parser 继续处理。
    """

    supported_suffixes = {".pdf", ".md", ".markdown", ".html", ".htm", ".txt"}
    text_suffixes = {".md", ".markdown", ".html", ".htm", ".txt"}

    def __init__(self, identity_builder: DocumentIdentityBuilder | None = None) -> None:
        self._identity_builder = identity_builder or DocumentIdentityBuilder()

    def load_directory(self, source_dir: Path) -> list[RawDocument]:
        """加载目录下所有支持的文档文件。

        目录不存在属于系统性输入错误，应该直接抛出异常；
        单个文件损坏则交给 IngestionPipeline 记录为文件级失败。
        """

        if not source_dir.exists():
            raise AppError(ErrorCode.DOCUMENT_LOAD_FAILED, f"文档目录不存在：{source_dir}")

        if not source_dir.is_dir():
            raise AppError(ErrorCode.DOCUMENT_LOAD_FAILED, f"文档来源不是目录：{source_dir}")

        documents: list[RawDocument] = []
        for path in self.iter_supported_files(source_dir):
            documents.append(self.load_file(path))
        return documents

    def iter_supported_files(self, source_dir: Path) -> Iterable[Path]:
        """按稳定顺序遍历支持的文件。"""

        for path in sorted(source_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in self.supported_suffixes:
                yield path

    def load_file(self, path: Path) -> RawDocument:
        """加载单个本地文件。"""

        try:
            raw_bytes = path.read_bytes()
        except OSError as exc:
            raise AppError(ErrorCode.DOCUMENT_LOAD_FAILED, f"文件读取失败：{path}") from exc

        suffix = path.suffix.lower()
        if suffix not in self.supported_suffixes:
            raise AppError(ErrorCode.DOCUMENT_LOAD_FAILED, f"不支持的文档类型：{path}")

        raw_text = ""
        if suffix in self.text_suffixes:
            try:
                raw_text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise AppError(ErrorCode.DOCUMENT_LOAD_FAILED, f"文件不是有效 UTF-8 文本：{path}") from exc

        doc_id = self._identity_builder.build_doc_id(path)
        content_hash = self._identity_builder.build_content_hash(raw_bytes)
        version_id = self._identity_builder.build_version_id(doc_id, content_hash)

        return RawDocument(
            doc_id=doc_id,
            source_path=str(path),
            source_uri=path.resolve().as_uri(),
            file_type=self._normalize_file_type(suffix),
            content_hash=content_hash,
            version_id=version_id,
            raw_text=raw_text,
            raw_bytes=raw_bytes,
            metadata={
                "filename": path.name,
                "suffix": suffix,
                "file_size": len(raw_bytes),
                "content_hash": content_hash,
                "version_id": version_id,
                "loaded_at": datetime.now(UTC).isoformat(),
            },
        )

    @staticmethod
    def _normalize_file_type(suffix: str) -> str:
        """把文件后缀归一化为内部文档类型。"""

        if suffix in {".md", ".markdown"}:
            return "markdown"
        if suffix in {".html", ".htm"}:
            return "html"
        if suffix == ".pdf":
            return "pdf"
        return suffix.lower().lstrip(".")


class LocalTextLoader(LocalDocumentLoader):
    """兼容子模块 1 的旧类名。

    新代码请优先使用 LocalDocumentLoader。保留这个类是为了让旧测试和旧入口继续工作。
    """

    supported_suffixes = {".md", ".markdown", ".txt"}
    text_suffixes = {".md", ".markdown", ".txt"}

    @staticmethod
    def _build_doc_id(path: Path) -> str:
        """兼容旧测试中的私有方法调用。"""

        return DocumentIdentityBuilder.build_doc_id(path)

    @staticmethod
    def _build_content_hash(text: str) -> str:
        """兼容旧测试中的私有方法调用。"""

        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _build_version_id(doc_id: str, content_hash: str) -> str:
        """兼容旧测试中的私有方法调用。"""

        return DocumentIdentityBuilder.build_version_id(doc_id, content_hash)


class StrictTextLoader(LocalDocumentLoader):
    """只加载文本类文件的 loader。

    TODO 子模块2-练习1：
    当前类直接继承 LocalDocumentLoader。请你为它补充一个测试：
    当目录中同时存在 .md、.html、.pdf 时，StrictTextLoader 应只返回 .md/.txt。
    """

    supported_suffixes = {".md", ".markdown", ".txt"}
    text_suffixes = {".md", ".markdown", ".txt"}
