"""Compare search 聚合报告 writer。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.retrieval.reporting.config import RetrievalReportConfig
from app.retrieval.reporting.models import RetrievalComparisonExecutionReport


class RetrievalComparisonReportWriter:
    """把一次 compare search 的汇总结果写成稳定 JSON。"""

    def write(
        self,
        report: RetrievalComparisonExecutionReport,
        output_path: Path,
        config: RetrievalReportConfig,
    ) -> Path:
        """写入报告并返回路径；调用方必须提前准备目录。"""

        _ = config
        output_path.write_text(
            json.dumps(
                self.build_report(report),
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return output_path

    @staticmethod
    def build_report(report: RetrievalComparisonExecutionReport) -> dict[str, Any]:
        """构建可 JSON 序列化的 compare search 聚合报告。"""

        return {
            "schema_version": 1,
            "report_type": "retrieval_comparison",
            "generated_at": report.generated_at,
            "trace_id": report.trace.trace_id,
            "status": report.status,
            "request": {
                "query": report.query,
                "top_k": report.top_k,
                "retrievers": list(report.retrievers),
            },
            "runtime": asdict(report.runtime),
            "strategy_results": [
                asdict(strategy_result) for strategy_result in report.strategy_results
            ],
            "overlaps": [asdict(overlap) for overlap in report.overlaps],
            "trace": {
                "final_status": report.trace.final_status,
                "failure_type": report.trace.failure_type,
                "error_message": report.trace.error_message,
                "latency_ms": report.trace.latency_ms,
                "stages": [asdict(stage) for stage in report.trace.stages],
            },
        }
