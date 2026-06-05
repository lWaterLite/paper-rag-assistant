"""API 路由契约设计。

这里暂时不引入 FastAPI，只维护一份可测试的路由规划。
后续真正接入服务端时，可以把 request_model 和 response_model 映射到 FastAPI 装饰器。
"""

from __future__ import annotations


def planned_routes() -> list[dict[str, str]]:
    """返回本项目后续计划实现的 API 路由。"""

    return [
        {
            "method": "GET",
            "path": "/health",
            "request_model": "None",
            "response_model": "HealthResponse",
            "description": "健康检查",
        },
        {
            "method": "POST",
            "path": "/documents/ingest",
            "request_model": "DocumentIngestRequest",
            "response_model": "DocumentIngestResponse",
            "description": "导入文档并构建索引",
        },
        {
            "method": "GET",
            "path": "/documents",
            "request_model": "None",
            "response_model": "DocumentListResponse",
            "description": "查看已导入文档",
        },
        {
            "method": "POST",
            "path": "/search",
            "request_model": "SearchRequest",
            "response_model": "SearchResponse",
            "description": "只执行检索，不生成回答",
        },
        {
            "method": "POST",
            "path": "/ask",
            "request_model": "AskRequest",
            "response_model": "AskResponse",
            "description": "执行 RAG 问答",
        },
    ]


def api_contract() -> dict[str, list[dict[str, str]] | dict[str, str]]:
    """返回 API 总体契约说明。

    trace_id 默认放在响应体中，方便日志、前端和命令行调用统一处理；
    真正接入 HTTP 后可以额外放入 X-Trace-Id 响应头。
    """

    return {
        "routes": planned_routes(),
        "error_response": {
            "response_model": "ErrorResponse",
            "description": "所有业务错误统一返回 code、message、trace_id 和可选 detail",
        },
    }


__all__ = ["api_contract", "planned_routes"]
