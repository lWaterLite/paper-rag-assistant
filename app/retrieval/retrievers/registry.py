"""检索策略注册表。"""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock

from app.retrieval.retrievers.base import Retriever

RetrieverProvider = Callable[[], Retriever]


class RetrieverRegistry:
    """按策略名惰性创建并缓存 Retriever。

    provider 由 factory 注册，因此 registry 不需要知道 RagIndex 或具体检索器依赖。
    同一策略在一个 registry 生命周期内只创建一次，hybrid 可以安全复用底层实例。
    """

    def __init__(self) -> None:
        self._providers: dict[str, RetrieverProvider] = {}
        self._instances: dict[str, Retriever] = {}
        self._resolving: set[str] = set()
        self._lock = RLock()

    def register(self, name: str, provider: RetrieverProvider) -> None:
        """注册一个检索策略 provider。"""

        if not isinstance(name, str):
            raise ValueError("retriever 策略名称必须是字符串")
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("retriever 策略名称不能为空")
        if not callable(provider):
            raise TypeError("retriever provider 必须可调用")

        with self._lock:
            if normalized_name in self._providers:
                raise ValueError(f"retriever 策略已注册：{normalized_name}")
            self._providers[normalized_name] = provider

    def resolve(self, name: str) -> Retriever:
        """解析策略，并在首次使用时创建对应 Retriever。"""

        if not isinstance(name, str):
            raise ValueError("retriever 策略名称必须是字符串")
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("retriever 策略名称不能为空")
        with self._lock:
            cached = self._instances.get(normalized_name)
            if cached is not None:
                return cached

            provider = self._providers.get(normalized_name)
            if provider is None:
                supported = ", ".join(self.list_strategies()) or "无"
                raise ValueError(
                    f"未知 retriever strategy：{name}，"
                    f"当前已注册策略：{supported}"
                )
            if normalized_name in self._resolving:
                raise RuntimeError(
                    f"retriever provider 存在循环依赖：{normalized_name}"
                )

            self._resolving.add(normalized_name)
            try:
                retriever = provider()
                if not callable(getattr(retriever, "retrieve", None)):
                    raise TypeError(
                        f"retriever provider 返回了无效对象：{normalized_name}"
                    )
                self._instances[normalized_name] = retriever
                return retriever
            finally:
                self._resolving.remove(normalized_name)

    def list_strategies(self) -> tuple[str, ...]:
        """返回已注册的策略名称。"""

        with self._lock:
            return tuple(sorted(self._providers))
