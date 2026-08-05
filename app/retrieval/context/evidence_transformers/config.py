"""候选证据变换的运行时配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EvidenceTransformFailureMode = Literal["fail_open", "fail_closed"]


@dataclass(frozen=True, slots=True)
class EvidenceTransformationConfig:
    """EvidenceTransformStage 实际接收的运行时配置。"""

    enabled: bool = True
    strategy: str = "passthrough"
    failure_mode: EvidenceTransformFailureMode = "fail_open"

    def __post_init__(self) -> None:
        normalized_strategy = self.strategy.strip()
        if not normalized_strategy:
            raise ValueError("evidence transformation strategy 不能为空")
        object.__setattr__(self, "strategy", normalized_strategy)
