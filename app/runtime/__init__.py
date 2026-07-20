"""应用运行期容器、生命周期管理与 Web 适配能力。"""

from app.runtime.application import ApplicationRuntime, ApplicationRuntimeState
from app.runtime.web import create_web_lifespan

__all__ = [
    "ApplicationRuntime",
    "ApplicationRuntimeState",
    "create_web_lifespan",
]
