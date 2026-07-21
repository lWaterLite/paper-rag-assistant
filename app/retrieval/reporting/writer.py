"""Retrieval JSON 报告 writer。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.retrieval.models import RetrievedChunk
from app.retrieval.reporting.config import RetrievalReportConfig
from app.retrieval.reporting.models import RetrievalExecutionReport


class RetrievalReportWriter:
    """把 retrieval 执行报告写成稳定 JSON。"""

    def write(
        self,
        report: RetrievalExecutionReport,
        output_path: Path,
        config: RetrievalReportConfig,
    ) -> Path:
        """写入报告并返回路径；调用方必须提前准备目录。"""

        output_path.write_text(
            json.dumps(
                self.build_report(report, config),
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return output_path

    @staticmethod
    def build_report(
        report: RetrievalExecutionReport,
        config: RetrievalReportConfig,
    ) -> dict[str, Any]:
        """构建可 JSON 序列化的 retrieval 报告。"""

        return {
            "schema_version": 1,
            "report_type": "retrieval_execution",
            "generated_at": report.generated_at,
            "trace_id": report.trace.trace_id,
            "status": report.trace.final_status,
            "request": {
                "query": report.query,
                "requested_top_k": report.requested_top_k,
                "resolved_top_k": report.resolved_top_k,
                "resolved_candidate_limit": report.resolved_candidate_limit,
                "requested_retriever": report.requested_retriever,
                "resolved_retriever": report.resolved_retriever,
            },
            "counts": {
                "candidate_count": report.candidate_count,
                "deduplicated_count": report.deduplicated_count,
                "returned_count": report.returned_count,
            },
            "runtime": asdict(report.runtime),
            "stages": [asdict(stage) for stage in report.stage_observations],
            "results": [
                _serialize_result(result, config) for result in report.results
            ],
            "failure": {
                "error_code": report.error_code,
                "error_message": report.error_message,
            }
            if report.error_code is not None or report.error_message is not None
            else None,
            "trace": {
                "final_status": report.trace.final_status,
                "failure_type": report.trace.failure_type,
                "error_message": report.trace.error_message,
                "latency_ms": report.trace.latency_ms,
                "stages": [asdict(stage) for stage in report.trace.stages],
            },
        }


def _serialize_result(
    result: RetrievedChunk,
    config: RetrievalReportConfig,
) -> dict[str, Any]:
    """序列化结果摘要，默认不把完整 chunk 文本写入日志。"""

    payload: dict[str, Any] = {
        "chunk_id": result.chunk_id,
        "doc_id": result.doc_id,
        "version_id": result.version_id,
        "rank": result.rank,
        "score": result.score,
        "retriever": result.retriever,
        "source_path": Path(result.source_path).as_posix(),
        "chunk_index": result.chunk_index,
        "title": result.title,
        "section": result.section,
        "page_start": result.page_start,
        "page_end": result.page_end,
        "retrieval_signals": [asdict(signal) for signal in result.retrieval_signals],
        "rerank_signal": (
            asdict(result.rerank_signal) if result.rerank_signal is not None else None
        ),
    }
    if config.include_result_text:
        payload["text_preview"] = result.text[: config.result_preview_chars]
    return payload
