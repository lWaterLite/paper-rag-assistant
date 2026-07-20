"""应用运行期容器、生命周期管理与 Web 适配能力。"""

from app.runtime.application import ApplicationRuntime
from app.runtime.base import Runtime, RuntimeState
from app.runtime.manager import RuntimeManager
from app.runtime.web import create_web_lifespan

__all__ = [
    "ApplicationRuntime",
    "Runtime",
    "RuntimeManager",
    "RuntimeState",
    "create_web_lifespan",
]
