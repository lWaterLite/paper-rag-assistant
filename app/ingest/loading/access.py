"""受限入口的本地文档目录访问策略。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from app.core.errors import AppError, ErrorCode


@dataclass(frozen=True)
class DocumentSourceAccessConfig:
    """文档导入入口允许使用的本地目录配置。"""

    allowed_source_dirs: tuple[Path, ...]

    def __post_init__(self) -> None:
        if not self.allowed_source_dirs:
            raise ValueError("allowed_source_dirs 至少需要一个目录")
        normalized_dirs = tuple(
            dict.fromkeys(
                path.expanduser().resolve(strict=False)
                for path in self.allowed_source_dirs
            )
        )
        object.__setattr__(self, "allowed_source_dirs", normalized_dirs)


class DocumentSourceAccessService:
    """解析并校验 API 等受限入口提交的文档目录。"""

    def __init__(self, config: DocumentSourceAccessConfig) -> None:
        self._config = config

    def resolve_directory(self, requested_source_dir: str | Path) -> Path:
        """返回位于白名单内、存在且可作为文档来源的规范化目录。"""

        if isinstance(requested_source_dir, str) and not requested_source_dir.strip():
            raise AppError(ErrorCode.DOCUMENT_LOAD_FAILED, "文档目录不能为空")

        requested_path = Path(requested_source_dir).expanduser()
        try:
            resolved_path = requested_path.resolve(strict=True)
        except OSError as exc:
            raise AppError(
                ErrorCode.DOCUMENT_LOAD_FAILED,
                f"文档目录不存在或无法解析：{requested_path}",
            ) from exc

        if not resolved_path.is_dir():
            raise AppError(
                ErrorCode.DOCUMENT_LOAD_FAILED,
                f"文档来源不是目录：{resolved_path}",
            )
        try:
            # 仅打开目录句柄，不读取或遍历其中的文档内容。
            with os.scandir(resolved_path):
                pass
        except OSError as exc:
            raise AppError(
                ErrorCode.DOCUMENT_LOAD_FAILED,
                f"文档目录不可读取：{resolved_path}",
            ) from exc
        if not any(
            _is_relative_to(resolved_path, allowed_dir)
            for allowed_dir in self._config.allowed_source_dirs
        ):
            allowed_dirs = ", ".join(
                path.as_posix() for path in self._config.allowed_source_dirs
            )
            raise AppError(
                ErrorCode.DOCUMENT_LOAD_FAILED,
                "文档目录不在允许的导入根目录内："
                f"requested={resolved_path.as_posix()}，allowed={allowed_dirs}",
            )
        return resolved_path


def _is_relative_to(path: Path, parent: Path) -> bool:
    """兼容地判断 path 是否位于 parent 之下。"""

    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
