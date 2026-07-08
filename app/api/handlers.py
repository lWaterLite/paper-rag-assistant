"""API 层可复用处理函数。"""

from __future__ import annotations

from app.api.schemas import SearchRequest, SearchResponse, retrieved_chunk_to_response, trace_to_response
from app.retrieval.service import SearchService


def handle_search_request(request: SearchRequest, search_service: SearchService) -> SearchResponse:
    """处理 /search 请求。

    这里不依赖具体 Web 框架，后续接入 FastAPI 时可以直接在 route 中调用。
    """

    result = search_service.search(
        request.query,
        top_k=request.top_k,
        retriever=request.retriever,
    )
    return SearchResponse(
        query=result.query,
        results=[retrieved_chunk_to_response(chunk) for chunk in result.results],
        trace_id=result.trace.trace_id,
        top_k=result.top_k,
        retriever=result.retriever,
        latency_ms=result.trace.latency_ms,
        trace=trace_to_response(result.trace) if request.debug_trace else None,
    )
