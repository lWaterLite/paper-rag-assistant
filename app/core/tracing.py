"""跨模块流程追踪模型。"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

StageStatus = Literal["success", "error"]
TraceFinalStatus = Literal["running", "success", "error"]


def new_id(prefix: str) -> str:
    """生成带前缀的短 ID，方便追踪与排查。"""

    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True)
class PipelineStageRun:
    """一次流程阶段的运行记录。"""

    stage: str
    status: StageStatus
    latency_ms: float
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class RagTrace:
    """一次跨模块流程的完整追踪信息。"""

    trace_id: str = field(default_factory=lambda: new_id("trace"))
    started_at: float = field(default_factory=time.perf_counter)
    stages: list[PipelineStageRun] = field(default_factory=list)
    final_status: TraceFinalStatus = "running"
    failure_type: str | None = None
    error_message: str | None = None

    def record_stage(
        self,
        stage: str,
        status: StageStatus,
        started_at: float,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """记录阶段耗时和摘要信息。"""

        self.stages.append(
            PipelineStageRun(
                stage=stage,
                status=status,
                latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
                detail=detail or {},
            )
        )

    def mark_success(self) -> None:
        """标记流程成功结束。"""

        self.final_status = "success"
        self.failure_type = None
        self.error_message = None

    def mark_failed(self, failure_type: str, error_message: str) -> None:
        """标记流程失败结束。"""

        self.final_status = "error"
        self.failure_type = failure_type
        self.error_message = error_message

    @property
    def latency_ms(self) -> float:
        """返回流程从开始到当前的耗时。"""

        return round((time.perf_counter() - self.started_at) * 1000, 2)
