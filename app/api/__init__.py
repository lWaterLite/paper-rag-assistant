"""API 层边界。

当前阶段不引入 FastAPI 依赖，但保留可测试的 schema、路由契约和 handler。
后续服务化时，可以把 handler 接入真实路由。
"""

from app.api.handlers import handle_search_request

__all__ = ["handle_search_request"]
