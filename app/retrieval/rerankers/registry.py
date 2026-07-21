"""重排序策略注册表。"""

from __future__ import annotations

from collections.abc import Callable

from app.retrieval.rerankers.base import Reranker
from app.retrieval.rerankers.config import RerankingConfig
from app.retrieval.tokenizers.base import Tokenizer

RerankerProvider = Callable[[RerankingConfig], Reranker]


class RerankerRegistry:
    """根据策略名称创建 reranker，不让 pipeline 依赖具体实现。"""

    def __init__(self) -> None:
        self._providers: dict[str, RerankerProvider] = {}

    def register(self, name: str, provider: RerankerProvider) -> None:
        """注册一个 reranker provider。"""

        normalized_name = _normalize_name(name)
        if not callable(provider):
            raise TypeError("reranker provider 必须可调用")
        if normalized_name in self._providers:
            raise ValueError(f"reranker 策略已注册：{normalized_name}")
        self._providers[normalized_name] = provider

    def create(self, config: RerankingConfig) -> Reranker:
        """根据运行时配置创建指定 reranker。"""

        provider = self._providers.get(config.strategy)
        if provider is None:
            supported = ", ".join(self.list_strategies()) or "无"
            raise ValueError(
                f"未知 reranker strategy：{config.strategy}，当前已注册策略：{supported}"
            )
        reranker = provider(config)
        if not callable(getattr(reranker, "rerank", None)):
            raise TypeError(
                f"reranker provider 返回了无效对象：{config.strategy}"
            )
        if not isinstance(getattr(reranker, "name", None), str):
            raise TypeError(
                f"reranker provider 缺少有效名称：{config.strategy}"
            )
        return reranker

    def list_strategies(self) -> tuple[str, ...]:
        """返回已注册策略名称。"""

        return tuple(sorted(self._providers))


def _normalize_name(name: str) -> str:
    """校验并规范化策略名称。"""

    if not isinstance(name, str):
        raise ValueError("reranker 策略名称必须是字符串")
    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError("reranker 策略名称不能为空")
    return normalized_name


def build_default_reranker_registry(tokenizer: Tokenizer) -> RerankerRegistry:
    """创建项目内置 reranker registry，并注入 lexical 所需 tokenizer。"""

    from app.retrieval.rerankers.lexical import LexicalReranker

    registry = RerankerRegistry()
    registry.register(
        "lexical",
        lambda config: LexicalReranker(tokenizer, batch_size=config.batch_size),
    )
    return registry
