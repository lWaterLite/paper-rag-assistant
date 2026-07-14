"""文档来源加载组件。"""

from app.ingest.loading.local import (
    DocumentIdentityBuilder,
    DocumentLoader,
    DocumentSource,
    LocalDocumentLoader,
    LocalDocumentLoaderConfig,
)

__all__ = [
    "DocumentIdentityBuilder",
    "DocumentLoader",
    "DocumentSource",
    "LocalDocumentLoader",
    "LocalDocumentLoaderConfig",
]
