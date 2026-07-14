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
from app.ingest.models import RawDocument


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

    def iter_supported_files(self, source_dir: Path) -> Iterable[Path]:
        """遍历目录中支持的文件。"""


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

        digest = hashlib.sha1(f"{doc_id}:{content_hash}".encode("utf-8")).hexdigest()[
            :12
        ]
        return f"v_{digest}"


@dataclass(frozen=True)
class LocalDocumentLoaderConfig:
    """本地文档扫描配置。

    默认值只存在于配置对象里。LocalDocumentLoader 必须显式接收这个配置对象，
    避免生产代码里出现一部分 loader 使用 env 配置、一部分 loader 使用隐式默认值。
    """

    recursive: bool = True

    ignored_dir_names: frozenset[str] = frozenset(
        {".git", ".idea", "__pycache__", ".tmp_tests"}
    )
    ignored_relative_paths: tuple[str, ...] = ("data/indexes",)
    skip_hidden_paths: bool = True
    temporary_file_prefixes: tuple[str, ...] = ("~$",)
    temporary_file_suffixes: tuple[str, ...] = (".tmp", ".part", ".crdownload")


class LocalDocumentLoader:
    """加载本地论文和文档文件。

    PDF 是二进制格式，loading 阶段只读取字节，不在这里做文本提取；
    Markdown、HTML、TXT 会同时保存 raw_text，方便 parser 继续处理。
    """

    supported_suffixes = {".pdf", ".md", ".markdown", ".html", ".htm", ".txt"}
    text_suffixes = {".md", ".markdown", ".html", ".htm", ".txt"}

    def __init__(
        self,
        config: LocalDocumentLoaderConfig,
        identity_builder: DocumentIdentityBuilder,
    ) -> None:
        self._config = config
        self._identity_builder = identity_builder

    def load_directory(self, source_dir: Path) -> list[RawDocument]:
        """加载目录下所有支持的文档文件。

        目录不存在属于系统性输入错误，应该直接抛出异常；
        单个文件损坏则交给 IngestionPipeline 记录为文件级失败。
        """

        if not source_dir.exists():
            raise AppError(
                ErrorCode.DOCUMENT_LOAD_FAILED, f"文档目录不存在：{source_dir}"
            )

        if not source_dir.is_dir():
            raise AppError(
                ErrorCode.DOCUMENT_LOAD_FAILED, f"文档来源不是目录：{source_dir}"
            )

        documents: list[RawDocument] = []
        for path in self.iter_supported_files(source_dir):
            documents.append(self.load_file(path))
        return documents

    def iter_supported_files(self, source_dir: Path) -> Iterable[Path]:
        """按稳定顺序遍历支持的文件。

        TODO 子模块2-练习1：
        请把这里升级成真实工程可用的目录扫描策略。
        需要支持：
        1. 可配置是否递归扫描子目录。
        2. 跳过隐藏目录和工程产物目录，例如 .git、.tmp_tests、__pycache__、data/indexes。
        3. 跳过临时文件，例如以 "~$" 开头或以 ".tmp" 结尾的文件。
        4. 在不改变 load_file 职责的前提下，保持输出路径顺序稳定。
        """

        globber = source_dir.rglob if self._config.recursive else source_dir.glob

        for path in sorted(globber("*"), key=lambda item: item.as_posix().lower()):
            if self._should_skip_path(path, source_dir):
                continue

            if path.is_file() and path.suffix.lower() in self.supported_suffixes:
                yield path

    def _should_skip_path(self, path: Path, source_dir: Path) -> bool:
        """判断扫描到的路径是否应该跳过。"""

        relative = path.relative_to(source_dir)

        if self._has_ignored_directory(relative, is_dir=path.is_dir()):
            return True

        if self._config.skip_hidden_paths and self._has_hidden_path_part(relative):
            return True

        if self._matches_ignored_relative_path(relative):
            return True

        if self._is_temporary_file(path):
            return True

        return False

    def _has_ignored_directory(self, relative: Path, *, is_dir: bool) -> bool:
        """判断路径中是否包含需要跳过的目录名。"""

        parts = relative.parts if is_dir else relative.parts[:-1]
        return any(part in self._config.ignored_dir_names for part in parts)

    @staticmethod
    def _has_hidden_path_part(relative: Path) -> bool:
        """判断路径中是否包含隐藏目录或隐藏文件。"""

        return any(part.startswith(".") for part in relative.parts)

    def _matches_ignored_relative_path(self, relative: Path) -> bool:
        """判断路径是否位于需要忽略的相对目录下。"""

        relative_posix = relative.as_posix().lower()
        ignored_paths = tuple(
            path.strip("/").lower() for path in self._config.ignored_relative_paths
        )

        return any(
            relative_posix == ignored_path
            or relative_posix.startswith(f"{ignored_path}/")
            for ignored_path in ignored_paths
            if ignored_path
        )

    def _is_temporary_file(self, path: Path) -> bool:
        """判断文件是否是常见临时文件。"""

        name = path.name.lower()
        return name.startswith(self._config.temporary_file_prefixes) or name.endswith(
            self._config.temporary_file_suffixes
        )

    def load_file(self, path: Path) -> RawDocument:
        """加载单个本地文件。"""

        try:
            raw_bytes = path.read_bytes()
        except OSError as exc:
            raise AppError(
                ErrorCode.DOCUMENT_LOAD_FAILED, f"文件读取失败：{path}"
            ) from exc

        suffix = path.suffix.lower()
        if suffix not in self.supported_suffixes:
            raise AppError(ErrorCode.DOCUMENT_LOAD_FAILED, f"不支持的文档类型：{path}")

        raw_text = ""
        if suffix in self.text_suffixes:
            try:
                raw_text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise AppError(
                    ErrorCode.DOCUMENT_LOAD_FAILED, f"文件不是有效 UTF-8 文本：{path}"
                ) from exc

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
