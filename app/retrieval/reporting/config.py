"""Retrieval 报告运行时配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RetrievalReportConfig:
    """Retrieval 报告生成与写入策略。"""

    enabled: bool = False
    output_dir: Path = Path("logs/retrieval")
    include_result_text: bool = False
    result_preview_chars: int = 160
    fail_on_write_error: bool = False

    def __post_init__(self) -> None:
        if self.result_preview_chars <= 0:
            raise ValueError("retrieval report result_preview_chars 必须大于 0")

    def output_path(self, trace_id: str) -> Path:
        """根据 trace_id 生成不会互相覆盖的报告路径。"""

        normalized_trace_id = trace_id.strip()
        if not normalized_trace_id:
            raise ValueError("retrieval report trace_id 不能为空")
        return self.output_dir / f"retrieval_{normalized_trace_id}.json"

    def comparison_output_path(self, trace_id: str) -> Path:
        """根据 compare search 的父 trace_id 生成聚合报告路径。"""

        normalized_trace_id = trace_id.strip()
        if not normalized_trace_id:
            raise ValueError("retrieval comparison report trace_id 不能为空")
        return self.output_dir / f"retrieval_comparison_{normalized_trace_id}.json"
