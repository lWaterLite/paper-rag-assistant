"""文档摄取阶段的内存集合。"""

from app.ingest.collections.documents import (
    DocumentCollection,
    InMemoryDocumentCollection,
)

__all__ = ["DocumentCollection", "InMemoryDocumentCollection"]
