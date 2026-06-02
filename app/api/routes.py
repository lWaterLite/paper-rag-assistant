"""API 路由设计占位。

这里不直接引入 FastAPI，是为了避免子模块 1 练习修改你的依赖环境。
"""

from __future__ import annotations


def planned_routes() -> list[dict[str, str]]:
    """返回本项目后续计划实现的 API 路由。"""

    return [
        {"method": "GET", "path": "/health", "description": "健康检查"},
        {"method": "POST", "path": "/documents/ingest", "description": "导入文档并构建索引"},
        {"method": "GET", "path": "/documents", "description": "查看已导入文档"},
        {"method": "POST", "path": "/search", "description": "只执行检索，不生成回答"},
        {"method": "POST", "path": "/ask", "description": "执行 RAG 问答"},
    ]

    # TODO 练习 13：
    # 请你为 /ask 和 /search 设计请求与响应 JSON。
    # 思考：
    # 1. 请求中是否应该包含 top_k？
    # 2. 响应中是否应该暴露 retrieved_chunks？
    # 3. trace_id 应该放在响应体里，还是响应 header 里？

