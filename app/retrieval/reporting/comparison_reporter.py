"""Compare search 聚合报告协调器。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.retrieval.reporting.config import RetrievalReportConfig
from app.retrieval.reporting.models import (
    RetrievalComparisonExecutionReport,
    RetrievalRuntimeSnapshot,
)
from app.retrieval.reporting.reporter import RetrievalReportWriteResult, RetrievalReporter
from app.retrieval.reporting.comparison_writer import RetrievalComparisonReportWriter


@dataclass(frozen=True, slots=True)
class RetrievalComparisonReporter:
    """持有 compare search 报告策略、运行时快照与 writer。"""

    config: RetrievalReportConfig
    runtime_snapshot: RetrievalRuntimeSnapshot
    writer: RetrievalComparisonReportWriter

    @classmethod
    def disabled(cls) -> "RetrievalComparisonReporter":
        """创建显式禁用的 compare search reporter，适合独立组件测试。"""

        base_reporter = RetrievalReporter.disabled()
        return cls(
            config=RetrievalReportConfig(enabled=False),
            runtime_snapshot=base_reporter.runtime_snapshot,
            writer=RetrievalComparisonReportWriter(),
        )

    @property
    def enabled(self) -> bool:
        """聚合报告功能是否启用。"""

        return self.config.enabled

    def prepare_output_directory(self) -> None:
        """在流程启动阶段准备报告目录。"""

        if self.config.enabled:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        report: RetrievalComparisonExecutionReport,
    ) -> RetrievalReportWriteResult:
        """按配置写入 compare search 聚合报告。"""

        if not self.config.enabled:
            return RetrievalReportWriteResult()

        output_path = self.config.comparison_output_path(report.trace.trace_id)
        try:
            path = self.writer.write(report, output_path, self.config)
            return RetrievalReportWriteResult(path=path)
        except OSError as exc:
            return RetrievalReportWriteResult(
                error_message=str(exc),
                fatal=self.config.fail_on_write_error,
            )
