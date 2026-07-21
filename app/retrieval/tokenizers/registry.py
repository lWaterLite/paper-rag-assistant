"""分词器策略注册表。"""

from __future__ import annotations

from collections.abc import Callable

from app.retrieval.tokenizers.base import Tokenizer
from app.retrieval.tokenizers.config import TokenizerConfig

TokenizerProvider = Callable[[], Tokenizer]


class TokenizerRegistry:
    """维护分词器策略名称与实例提供者之间的映射。"""

    def __init__(self) -> None:
        self._providers: dict[str, TokenizerProvider] = {}

    def register(self, name: str, provider: TokenizerProvider) -> None:
        """注册分词器实例提供者。"""

        if not isinstance(name, str):
            raise ValueError("tokenizer 策略名称必须是字符串")
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("tokenizer 策略名称不能为空")
        if normalized_name in self._providers:
            raise ValueError(f"tokenizer 策略已注册：{normalized_name}")
        if not callable(provider):
            raise TypeError("tokenizer provider 必须可调用")
        self._providers[normalized_name] = provider

    def create(self, config: TokenizerConfig) -> Tokenizer:
        """根据运行时配置创建分词器。"""

        provider = self._providers.get(config.strategy)
        if provider is None:
            supported_strategies = ", ".join(self.list_strategies()) or "无"
            raise ValueError(
                f"未知 tokenizer strategy：{config.strategy}，"
                f"当前已注册策略：{supported_strategies}"
            )
        return provider()

    def list_strategies(self) -> tuple[str, ...]:
        """返回所有已注册策略名称。"""

        return tuple(sorted(self._providers))


def build_default_tokenizer_registry() -> TokenizerRegistry:
    """创建包含项目内置策略的分词器注册表。"""

    from app.retrieval.tokenizers.regex import RegexTokenizer

    registry = TokenizerRegistry()
    registry.register("regex", RegexTokenizer)
    return registry
