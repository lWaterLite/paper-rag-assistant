"""API schema 占位。

当前直接复用核心模型。后续接入 FastAPI 时，可以把这些 dataclass 替换或映射为 Pydantic model。
"""

from __future__ import annotations

from app.core.models import Citation, RagAnswer, RetrievedChunk

__all__ = ["Citation", "RagAnswer", "RetrievedChunk"]

