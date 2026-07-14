"""API 共享契约与边界校验工具。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    """所有 API 契约的基础模型。"""

    model_config = ConfigDict(extra="forbid")


def ensure_not_blank(value: str, field_name: str) -> str:
    """校验字符串不是空白内容，并返回去除首尾空白后的值。"""

    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} 不能为空")
    return cleaned


class TraceStageResponse(ApiModel):
    """单个流程阶段的追踪信息。"""

    stage: str
    status: Literal["success", "error"]
    latency_ms: float
    detail: dict[str, Any] = Field(default_factory=dict)


class TraceResponse(ApiModel):
    """一次请求的完整追踪信息。"""

    trace_id: str
    final_status: Literal["running", "success", "error"]
    latency_ms: float
    failure_type: str | None = None
    error_message: str | None = None
    stages: list[TraceStageResponse] = Field(default_factory=list)


class HealthResponse(ApiModel):
    """健康检查响应体。"""

    status: Literal["ok"] = "ok"
    service: str = "paper-rag-assistant"


class ErrorResponse(ApiModel):
    """统一业务错误响应体。"""

    code: str
    message: str
    trace_id: str | None = None
    detail: dict[str, Any] | None = None
