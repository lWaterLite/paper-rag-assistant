"""与具体 Web 框架解耦的 Runtime 生命周期适配器。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from app.runtime.manager import RuntimeManager


def create_web_lifespan(
    manager: RuntimeManager,
) -> Callable[[object], AbstractAsyncContextManager[None]]:
    """创建兼容 FastAPI lifespan 约定的异步上下文工厂。

    返回值可直接作为 FastAPI 的 `lifespan` 参数，也可以被其他采用
    “接收应用对象、返回异步上下文管理器”约定的 Web 框架复用。本模块不导入
    FastAPI，因此 Runtime 层不会依赖具体 Web 框架。
    """

    @asynccontextmanager
    async def lifespan(_: object) -> AsyncIterator[None]:
        async with manager.lifespan():
            yield

    return lifespan
