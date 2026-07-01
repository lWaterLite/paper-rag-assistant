"""结构化 metadata 基础能力。

本模块只放跨领域通用能力，不包含 chunking、retrieval 等具体业务语义。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BaseMetadata:
    """结构化 metadata 基类。

    子类负责声明具体字段；基类只提供通用序列化能力。
    """

    def to_dict(self, *, exclude_none: bool = True) -> dict[str, Any]:
        """转换成可放入模型 metadata 字段的字典。"""

        data = asdict(self)
        if not exclude_none:
            return data
        return {key: value for key, value in data.items() if value is not None}
