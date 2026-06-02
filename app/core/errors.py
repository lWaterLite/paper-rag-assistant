"""统一错误类型。

RAG 的每个阶段都可能失败。显式错误类型能帮助我们判断问题来自解析、索引、检索还是生成。
"""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    """当前练习中会用到的错误码。"""

    INVALID_CONFIG = "INVALID_CONFIG"
    DOCUMENT_LOAD_FAILED = "DOCUMENT_LOAD_FAILED"
    DOCUMENT_PARSE_FAILED = "DOCUMENT_PARSE_FAILED"
    CHUNK_FAILED = "CHUNK_FAILED"
    INDEX_FAILED = "INDEX_FAILED"
    RETRIEVAL_FAILED = "RETRIEVAL_FAILED"
    GENERATION_FAILED = "GENERATION_FAILED"


class AppError(Exception):
    """应用内部统一异常。"""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

