"""Repository 注册表共享基础能力。"""

from __future__ import annotations

from typing import TypeVar

BuilderT = TypeVar("BuilderT")


class RepositoryRegistryBase[BuilderT]:
    """按持久化类型保存领域 Repository 的构造器。

    各领域 Registry 负责声明自己的构造参数与返回协议；本类只统一处理注册、
    重复注册和未知类型错误，避免这些基础规则在多个 Registry 中漂移。
    """

    def __init__(self, *, subject: str) -> None:
        self._subject = subject
        self._builders: dict[str, BuilderT] = {}

    def register(
        self,
        repository_type: str,
        builder: BuilderT,
        *,
        replace: bool = False,
    ) -> None:
        """注册一种持久化实现的构造器。"""

        normalized_type = self._normalize_type(repository_type)
        if normalized_type in self._builders and not replace:
            raise ValueError(f"{self._subject} 已注册：{normalized_type}")
        self._builders[normalized_type] = builder

    def resolve(self, repository_type: str) -> BuilderT:
        """获取指定持久化类型的构造器。"""

        normalized_type = self._normalize_type(repository_type)
        builder = self._builders.get(normalized_type)
        if builder is None:
            available = ", ".join(sorted(self._builders)) or "无"
            raise ValueError(
                f"不支持的 {self._subject} 类型：{normalized_type}；已注册：{available}"
            )
        return builder

    @staticmethod
    def _normalize_type(repository_type: str) -> str:
        """规范类型名并拒绝空白值。"""

        normalized_type = repository_type.strip().lower()
        if not normalized_type:
            raise ValueError("repository 类型不能为空")
        return normalized_type
