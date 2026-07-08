"""chunker 策略注册表。"""

from __future__ import annotations

from app.ingest.chunking.strategies import (
    CharacterChunker,
    Chunker,
    ChunkerConfig,
    FixedTokenChunker,
    SectionAwareChunker,
)


class ChunkerRegistry:
    """chunker 策略注册表。

    registry 负责维护“策略名称 -> chunker 类”的映射。
    调用方只需要根据配置请求创建 chunker，不需要知道具体有哪些实现类。
    """

    def __init__(self) -> None:
        self._chunker_classes: dict[str, type[Chunker]] = {}

    def register(self, name: str, chunker_class: type[Chunker]) -> None:
        """注册一个 chunker 策略。"""

        if not isinstance(name, str):
            raise ValueError("chunker 策略名称必须是字符串")
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("chunker 策略名称不能为空")
        if normalized_name in self._chunker_classes:
            raise ValueError(f"chunker 策略已注册：{normalized_name}")
        if not isinstance(chunker_class, type) or not issubclass(
            chunker_class, Chunker
        ):
            raise TypeError("chunker_class 必须是 Chunker 的子类")
        self._chunker_classes[normalized_name] = chunker_class

    def validate_strategy(self, strategy: str) -> None:
        """校验策略名称是否已经注册。"""

        normalized_strategy = strategy.strip()
        if normalized_strategy not in self._chunker_classes:
            supported_strategies = ", ".join(self.list_strategies()) or "无"
            raise ValueError(
                f"未知 chunking strategy：{strategy}，"
                f"当前已注册策略：{supported_strategies}"
            )

    def create(self, config: ChunkerConfig) -> Chunker:
        """根据配置创建 chunker。"""

        self.validate_strategy(config.strategy)
        return self._chunker_classes[config.strategy.strip()](config)

    def list_strategies(self) -> tuple[str, ...]:
        """返回已注册策略名称，方便调试和测试。"""

        return tuple(sorted(self._chunker_classes))


def build_default_chunker_registry() -> ChunkerRegistry:
    """创建项目内置 chunker 策略注册表。"""

    registry = ChunkerRegistry()
    registry.register("character", CharacterChunker)
    registry.register("fixed_token", FixedTokenChunker)
    registry.register("section_aware", SectionAwareChunker)
    return registry
