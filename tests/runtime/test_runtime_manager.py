"""RuntimeManager 与 Web 生命周期适配器测试。"""

from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass, field

from app.runtime import RuntimeManager, RuntimeState, create_web_lifespan


@dataclass
class RecordingRuntime:
    """记录生命周期调用顺序的测试 Runtime。"""

    runtime_name: str
    events: list[str]
    fail_on_start: bool = False
    _state: RuntimeState = field(default=RuntimeState.CREATED, init=False)

    @property
    def name(self) -> str:
        return self.runtime_name

    @property
    def state(self) -> RuntimeState:
        return self._state

    def start(self) -> None:
        self.events.append(f"start:{self.name}")
        if self.fail_on_start:
            self._state = RuntimeState.FAILED
            raise RuntimeError(f"启动失败：{self.name}")
        self._state = RuntimeState.RUNNING

    def shutdown(self) -> None:
        self.events.append(f"shutdown:{self.name}")
        self._state = RuntimeState.STOPPED


class RuntimeManagerTest(unittest.TestCase):
    """验证多 Runtime 的生命周期顺序和 Web 适配。"""

    def test_manager_starts_in_order_and_shuts_down_in_reverse_order(self) -> None:
        events: list[str] = []
        manager = RuntimeManager()
        manager.register(RecordingRuntime("storage", events))
        manager.register(RecordingRuntime("application", events))

        manager.start()
        manager.shutdown()

        self.assertEqual(
            events,
            [
                "start:storage",
                "start:application",
                "shutdown:application",
                "shutdown:storage",
            ],
        )
        self.assertEqual(manager.state, RuntimeState.STOPPED)

    def test_manager_releases_started_runtimes_when_later_startup_fails(self) -> None:
        events: list[str] = []
        manager = RuntimeManager()
        manager.register(RecordingRuntime("storage", events))
        manager.register(RecordingRuntime("application", events, fail_on_start=True))

        with self.assertRaisesRegex(RuntimeError, "启动失败"):
            manager.start()

        self.assertEqual(events, ["start:storage", "start:application", "shutdown:storage"])
        self.assertEqual(manager.state, RuntimeState.FAILED)

    def test_web_lifespan_adapter_uses_runtime_manager(self) -> None:
        events: list[str] = []
        manager = RuntimeManager()
        manager.register(RecordingRuntime("application", events))
        lifespan = create_web_lifespan(manager)

        async def run_lifespan() -> None:
            async with lifespan(object()):
                self.assertEqual(manager.state, RuntimeState.RUNNING)

        asyncio.run(run_lifespan())

        self.assertEqual(events, ["start:application", "shutdown:application"])
        self.assertEqual(manager.state, RuntimeState.STOPPED)


if __name__ == "__main__":
    unittest.main()
