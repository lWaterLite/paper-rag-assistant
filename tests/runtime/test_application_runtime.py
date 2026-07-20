"""ApplicationRuntime 测试。"""

from __future__ import annotations

import asyncio
import shutil
import unittest
import uuid
from pathlib import Path

from app.core.settings import (
    EnvSettings,
    IndexingSettings,
    ProjectSettings,
    VectorRepositorySettings,
)
from app.factory import ApplicationFactory
from app.runtime import ApplicationRuntimeState, create_web_lifespan


class ApplicationRuntimeTest(unittest.TestCase):
    """验证在线服务由 Runtime 统一加载、复用与释放。"""

    def test_runtime_reuses_online_objects_until_shutdown(self) -> None:
        index_dir = Path(".tmp_tests") / f"application_runtime_{uuid.uuid4().hex}"
        try:
            factory = _create_factory(index_dir)
            _, build_result = factory.build_index_builder().build_from_directory(
                Path("data/raw/papers")
            )
            runtime = factory.build_runtime()

            self.assertEqual(runtime.state, ApplicationRuntimeState.CREATED)
            with self.assertRaisesRegex(RuntimeError, "尚未运行"):
                _ = runtime.search_service

            runtime.start()

            self.assertEqual(runtime.state, ApplicationRuntimeState.RUNNING)
            self.assertEqual(runtime.index.manifest.index_id, build_result.manifest.index_id)
            self.assertIs(runtime.search_service, runtime.search_service)
            self.assertIs(runtime.compare_search_service, runtime.compare_search_service)
            self.assertIs(runtime.rag_pipeline, runtime.rag_pipeline)

            result = runtime.search_service.search("RAG citation", top_k=2)
            self.assertGreater(len(result.results), 0)

            runtime.shutdown()

            self.assertEqual(runtime.state, ApplicationRuntimeState.STOPPED)
            with self.assertRaisesRegex(RuntimeError, "尚未运行"):
                _ = runtime.rag_pipeline
        finally:
            shutil.rmtree(index_dir, ignore_errors=True)

    def test_runtime_marks_failed_when_persisted_index_cannot_be_loaded(self) -> None:
        index_dir = Path(".tmp_tests") / f"missing_runtime_index_{uuid.uuid4().hex}"
        runtime = _create_factory(index_dir).build_runtime()

        with self.assertRaises(FileNotFoundError):
            runtime.start()

        self.assertEqual(runtime.state, ApplicationRuntimeState.FAILED)
        runtime.shutdown()
        self.assertEqual(runtime.state, ApplicationRuntimeState.STOPPED)

    def test_web_lifespan_adapter_manages_application_runtime(self) -> None:
        index_dir = Path(".tmp_tests") / f"web_lifespan_{uuid.uuid4().hex}"
        try:
            factory = _create_factory(index_dir)
            factory.build_index_builder().build_from_directory(Path("data/raw/papers"))
            runtime = factory.build_runtime()
            lifespan = create_web_lifespan(runtime)

            async def run_lifespan() -> None:
                async with lifespan(object()):
                    self.assertEqual(runtime.state, ApplicationRuntimeState.RUNNING)
                    self.assertGreater(
                        len(runtime.search_service.search("RAG citation", top_k=1).results),
                        0,
                    )

            asyncio.run(run_lifespan())

            self.assertEqual(runtime.state, ApplicationRuntimeState.STOPPED)
        finally:
            shutil.rmtree(index_dir, ignore_errors=True)


def _create_factory(index_dir: Path) -> ApplicationFactory:
    """创建使用隔离本地索引目录的 ApplicationFactory。"""

    return ApplicationFactory(
        env_settings=EnvSettings(),
        project_settings=ProjectSettings(
            indexing=IndexingSettings(
                vector_repository=VectorRepositorySettings(
                    type="local_json",
                    index_dir=index_dir,
                    collection_name="papers_test",
                )
            )
        ),
    )


if __name__ == "__main__":
    unittest.main()
