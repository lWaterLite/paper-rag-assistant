"""回答生成阶段的运行期 Config。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """控制回答 prompt、输出与引用失败语义的运行期配置。"""

    model: str
    temperature: float
    max_output_tokens: int
    timeout_seconds: float
    prompt_version: str
    default_language: str
    invalid_output_mode: Literal["fail_closed", "abstain"]

    def __post_init__(self) -> None:
        model = self.model.strip()
        prompt_version = self.prompt_version.strip()
        default_language = self.default_language.strip()
        if not model:
            raise ValueError("generation model 不能为空")
        if not 0 <= self.temperature <= 2:
            raise ValueError("generation temperature 必须在 0 到 2 之间")
        if self.max_output_tokens <= 0:
            raise ValueError("generation max_output_tokens 必须大于 0")
        if self.timeout_seconds <= 0:
            raise ValueError("generation timeout_seconds 必须大于 0")
        if not prompt_version:
            raise ValueError("generation prompt_version 不能为空")
        if not default_language:
            raise ValueError("generation default_language 不能为空")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "prompt_version", prompt_version)
        object.__setattr__(self, "default_language", default_language)


@dataclass(frozen=True, slots=True)
class CitationValidationConfig:
    """回答级 citation 校验的运行期配置。"""

    require_citations_when_evidence: bool = True
    require_inline_ids_match_payload: bool = True
    require_abstention_reason: bool = True
