"""应用对象组装入口。

这个包是项目的 composition root。底层对象不自己创建依赖，而是在这里统一组装。
调用方应显式创建 ApplicationFactory，让同一组 settings 在当前进程中保持一致。
"""

from __future__ import annotations

from app.factory.application import ApplicationFactory
__all__ = ["ApplicationFactory"]
