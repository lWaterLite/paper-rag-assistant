"""文档来源加载组件。"""

from app.ingest.loading.access import (
    DocumentSourceAccessConfig,
    DocumentSourceAccessService,
)
from app.ingest.loading.local import (
    DocumentLoader,
    DocumentSource,
)

__all__ = [
    "DocumentLoader",
    "DocumentSource",
    "DocumentSourceAccessConfig",
    "DocumentSourceAccessService",
]
