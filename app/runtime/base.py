"""运行时生命周期的通用协议与状态。"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol


class RuntimeState(StrEnum):
    """Runtime 的生命周期状态。"""

    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class Runtime(Protocol):
    """可被 RuntimeManager 管理的运行期组件。"""

    @property
    def name(self) -> str:
        """Runtime 的进程内唯一名称。"""

    @property
    def state(self) -> RuntimeState:
        """当前生命周期状态。"""

    def start(self) -> None:
        """启动并准备运行期资源。"""

    def shutdown(self) -> None:
        """释放运行期资源。"""
