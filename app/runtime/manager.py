"""多个 Runtime 的生命周期管理器。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator

from app.runtime.base import Runtime, RuntimeState


@dataclass(slots=True)
class RuntimeManager:
    """按注册顺序启动、按逆序关闭多个 Runtime。"""

    _runtimes: list[Runtime] = field(default_factory=list)
    _state: RuntimeState = field(default=RuntimeState.CREATED, init=False)

    @property
    def state(self) -> RuntimeState:
        """返回管理器的生命周期状态。"""

        return self._state

    def register(self, runtime: Runtime) -> None:
        """注册一个由当前管理器拥有生命周期的 Runtime。"""

        if self._state != RuntimeState.CREATED:
            raise RuntimeError("RuntimeManager 启动后不能再注册 Runtime")
        if any(item.name == runtime.name for item in self._runtimes):
            raise ValueError(f"Runtime 已注册：{runtime.name}")
        self._runtimes.append(runtime)

    def get(self, name: str) -> Runtime:
        """按名称读取已注册的 Runtime。"""

        for runtime in self._runtimes:
            if runtime.name == name:
                return runtime
        raise KeyError(f"未注册的 Runtime：{name}")

    def start(self) -> None:
        """启动全部 Runtime；失败时逆序回收已启动对象。"""

        if self._state == RuntimeState.RUNNING:
            return
        if self._state in {RuntimeState.STARTING, RuntimeState.STOPPING}:
            raise RuntimeError(f"RuntimeManager 当前不能启动：{self._state}")

        self._state = RuntimeState.STARTING
        started: list[Runtime] = []
        try:
            for runtime in self._runtimes:
                runtime.start()
                started.append(runtime)
        except Exception:
            for runtime in reversed(started):
                runtime.shutdown()
            self._state = RuntimeState.FAILED
            raise
        self._state = RuntimeState.RUNNING

    def shutdown(self) -> None:
        """按依赖反向顺序关闭全部 Runtime。"""

        if self._state == RuntimeState.STOPPED:
            return
        if self._state == RuntimeState.STOPPING:
            return

        self._state = RuntimeState.STOPPING
        errors: list[Exception] = []
        for runtime in reversed(self._runtimes):
            try:
                runtime.shutdown()
            except Exception as exc:
                errors.append(exc)
        self._state = RuntimeState.STOPPED
        if errors:
            raise RuntimeError("关闭 Runtime 时发生错误") from errors[0]

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[None]:
        """提供可供异步 Web 框架复用的生命周期上下文。"""

        self.start()
        try:
            yield
        finally:
            self.shutdown()
