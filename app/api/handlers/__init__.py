"""框架无关的 API 请求处理器。"""

from app.api.handlers.retrieval import (
    handle_ask_request,
    handle_compare_search_request,
    handle_search_request,
)

__all__ = [
    "handle_ask_request",
    "handle_compare_search_request",
    "handle_search_request",
]
