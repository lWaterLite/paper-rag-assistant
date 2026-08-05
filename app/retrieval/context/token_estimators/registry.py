"""Token estimator 策略注册表。"""

from __future__ import annotations

from collections.abc import Callable

from app.retrieval.context.token_estimators.base import TokenEstimator
from app.retrieval.context.token_estimators.config import TokenEstimatorConfig

TokenEstimatorProvider = Callable[[], TokenEstimator]


class TokenEstimatorRegistry:
    """根据策略名称创建 token estimator。"""

    def __init__(self) -> None:
        self._providers: dict[str, TokenEstimatorProvider] = {}

    def register(self, name: str, provider: TokenEstimatorProvider) -> None:
        """注册一个 token estimator provider。"""

        if not isinstance(name, str):
            raise ValueError("token estimator 策略名称必须是字符串")
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("token estimator 策略名称不能为空")
        if not callable(provider):
            raise TypeError("token estimator provider 必须可调用")
        if normalized_name in self._providers:
            raise ValueError(f"token estimator 策略已注册：{normalized_name}")
        self._providers[normalized_name] = provider

    def create(self, config: TokenEstimatorConfig) -> TokenEstimator:
        """根据配置创建 token estimator。"""

        provider = self._providers.get(config.strategy)
        if provider is None:
            supported = ", ".join(self.list_strategies()) or "无"
            raise ValueError(
                f"未知 token estimator strategy：{config.strategy}，"
                f"当前已注册策略：{supported}"
            )
        estimator = provider()
        if not callable(getattr(estimator, "count_text", None)):
            raise TypeError(
                f"token estimator provider 返回了无效对象：{config.strategy}"
            )
        if not isinstance(getattr(estimator, "name", None), str):
            raise TypeError(f"token estimator provider 缺少有效名称：{config.strategy}")
        return estimator

    def list_strategies(self) -> tuple[str, ...]:
        """返回已注册策略名称。"""

        return tuple(sorted(self._providers))


def build_default_token_estimator_registry() -> TokenEstimatorRegistry:
    """创建包含项目内置策略的 token estimator registry。"""

    from app.retrieval.context.token_estimators.regex import RegexTokenEstimator

    registry = TokenEstimatorRegistry()
    registry.register("regex", RegexTokenEstimator)
    return registry
