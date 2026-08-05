"""候选证据变换策略注册表。"""

from __future__ import annotations

from collections.abc import Callable

from app.retrieval.context.evidence_transformers.base import EvidenceTransformer
from app.retrieval.context.evidence_transformers.config import (
    EvidenceTransformationConfig,
)


EvidenceTransformerProvider = Callable[
    [EvidenceTransformationConfig], EvidenceTransformer
]


class EvidenceTransformerRegistry:
    """根据策略名称创建候选证据变换器。"""

    def __init__(self) -> None:
        self._providers: dict[str, EvidenceTransformerProvider] = {}

    def register(self, name: str, provider: EvidenceTransformerProvider) -> None:
        """注册 transformer provider。"""

        normalized_name = _normalize_name(name)
        if not callable(provider):
            raise TypeError("evidence transformer provider 必须可调用")
        if normalized_name in self._providers:
            raise ValueError(f"evidence transformer 策略已注册：{normalized_name}")
        self._providers[normalized_name] = provider

    def create(self, config: EvidenceTransformationConfig) -> EvidenceTransformer:
        """根据配置创建 transformer 并验证最小协议。"""

        provider = self._providers.get(config.strategy)
        if provider is None:
            supported = ", ".join(self.list_strategies()) or "无"
            raise ValueError(
                "未知 evidence transformer strategy："
                f"{config.strategy}，当前已注册策略：{supported}"
            )
        transformer = provider(config)
        if not callable(getattr(transformer, "transform", None)):
            raise TypeError(
                f"evidence transformer provider 返回了无效对象：{config.strategy}"
            )
        if not isinstance(getattr(transformer, "name", None), str):
            raise TypeError(
                f"evidence transformer provider 缺少有效名称：{config.strategy}"
            )
        return transformer

    def list_strategies(self) -> tuple[str, ...]:
        """返回已注册策略名称。"""

        return tuple(sorted(self._providers))


def build_default_evidence_transformer_registry() -> EvidenceTransformerRegistry:
    """创建包含项目内置 transformer 的注册表。"""

    from app.retrieval.context.evidence_transformers.passthrough import (
        PassthroughEvidenceTransformer,
    )

    registry = EvidenceTransformerRegistry()
    registry.register("passthrough", lambda config: PassthroughEvidenceTransformer())
    return registry


def _normalize_name(name: str) -> str:
    """校验并规范化策略名称。"""

    if not isinstance(name, str):
        raise ValueError("evidence transformer 策略名称必须是字符串")
    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError("evidence transformer 策略名称不能为空")
    return normalized_name
