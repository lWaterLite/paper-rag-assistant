"""在线 RAG 应用的运行期容器。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from threading import RLock
from typing import TYPE_CHECKING

from app.generation.answering import AnswerGenerator
from app.indexing.pipeline.types import RagIndex
from app.pipeline import RagPipeline
from app.retrieval.retrievers import RetrieverRegistry
from app.retrieval.services.search import CompareSearchService, SearchService

if TYPE_CHECKING:
    from app.factory.application import ApplicationFactory


class ApplicationRuntimeState(StrEnum):
    """Application Runtime 的生命周期状态。"""

    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(slots=True)
class ApplicationRuntime:
    """持有在线请求复用的 RAG 运行期对象。

    Factory 只负责组装对象；本类只负责在应用启动时加载一次持久化索引，构造并
    缓存在线服务。API Handler 应通过依赖注入取得这些服务，而不应自行创建
    Factory 或从磁盘重复加载索引。
    """

    factory: ApplicationFactory
    retriever_registry: RetrieverRegistry | None = None
    answer_generator: AnswerGenerator | None = None
    _state: ApplicationRuntimeState = field(
        default=ApplicationRuntimeState.CREATED,
        init=False,
    )
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)
    _index: RagIndex | None = field(default=None, init=False, repr=False)
    _search_service: SearchService | None = field(default=None, init=False, repr=False)
    _compare_search_service: CompareSearchService | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _rag_pipeline: RagPipeline | None = field(default=None, init=False, repr=False)

    @property
    def state(self) -> ApplicationRuntimeState:
        """返回当前 Runtime 状态。"""

        with self._lock:
            return self._state

    @property
    def index(self) -> RagIndex:
        """返回已加载的在线索引。"""

        with self._lock:
            return self._require_running("index", self._index)

    @property
    def search_service(self) -> SearchService:
        """返回请求间复用的检索服务。"""

        with self._lock:
            return self._require_running("search_service", self._search_service)

    @property
    def compare_search_service(self) -> CompareSearchService:
        """返回请求间复用的策略比较服务。"""

        with self._lock:
            return self._require_running(
                "compare_search_service",
                self._compare_search_service,
            )

    @property
    def rag_pipeline(self) -> RagPipeline:
        """返回请求间复用的 RAG 问答 Pipeline。"""

        with self._lock:
            return self._require_running("rag_pipeline", self._rag_pipeline)

    def start(self) -> None:
        """加载持久化索引并一次性初始化在线服务。"""

        with self._lock:
            if self._state == ApplicationRuntimeState.RUNNING:
                return
            if self._state in {
                ApplicationRuntimeState.STARTING,
                ApplicationRuntimeState.STOPPING,
            }:
                raise RuntimeError(f"ApplicationRuntime 当前不能启动：{self._state}")

            self._state = ApplicationRuntimeState.STARTING
            try:
                index = self.factory.build_rag_index_from_storage()
                search_service = self.factory.build_search_service(
                    index,
                    retriever_registry=self.retriever_registry,
                )
                compare_search_service = self.factory.build_compare_search_service(
                    index,
                    retriever_registry=self.retriever_registry,
                )
                rag_pipeline = self.factory.build_rag_pipeline(
                    index,
                    retriever_registry=self.retriever_registry,
                    answer_generator=self.answer_generator,
                )
            except Exception:
                self._clear_services()
                self._state = ApplicationRuntimeState.FAILED
                raise

            self._index = index
            self._search_service = search_service
            self._compare_search_service = compare_search_service
            self._rag_pipeline = rag_pipeline
            self._state = ApplicationRuntimeState.RUNNING

    def shutdown(self) -> None:
        """清理在线服务引用，为未来外部资源释放保留统一出口。"""

        with self._lock:
            if self._state == ApplicationRuntimeState.STOPPED:
                return
            if self._state == ApplicationRuntimeState.STOPPING:
                return

            self._state = ApplicationRuntimeState.STOPPING
            self._clear_services()
            self._state = ApplicationRuntimeState.STOPPED

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[ApplicationRuntime]:
        """提供可直接交给 Web 框架适配层的异步生命周期上下文。"""

        self.start()
        try:
            yield self
        finally:
            self.shutdown()

    def _clear_services(self) -> None:
        """清空由 Runtime 管理的在线对象。"""

        self._rag_pipeline = None
        self._compare_search_service = None
        self._search_service = None
        self._index = None

    def _require_running[T](self, name: str, value: T | None) -> T:
        """确保调用方不会在 Runtime 未启动时取得内部对象。"""

        if self._state != ApplicationRuntimeState.RUNNING or value is None:
            raise RuntimeError(
                f"ApplicationRuntime 尚未运行，不能读取 {name}；当前状态：{self._state}"
            )
        return value
