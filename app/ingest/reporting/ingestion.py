"""文档摄取报告输出。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.ingest.pipeline_types import IngestedDocument, IngestionResult


class IngestionReportWriter:
    """把摄取结果转换为可观察、可排查的 JSON 报告。"""

    def write(self, result: IngestionResult, output_path: Path) -> Path:
        """写入摄取报告并返回实际写入路径。"""

        output_path.write_text(
            json.dumps(self.build_report(result), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return output_path

    @staticmethod
    def build_report(result: IngestionResult) -> dict[str, Any]:
        """构建可 JSON 序列化的摄取报告。"""

        source_dir = result.metadata.get("source_dir", "")
        return {
            "trace_id": result.trace.trace_id,
            "source_dir": IngestionReportWriter._normalize_path(source_dir),
            "succeeded": len(result.documents),
            "failed": len(result.failures),
            "success": len(result.failures) == 0,
            "candidate_files": result.metadata.get(
                "candidate_files", len(result.documents) + len(result.failures)
            ),
            "documents": [
                IngestionReportWriter._serialize_document(document)
                for document in result.documents
            ],
            "failures": [asdict(failure) for failure in result.failures],
            "trace": {
                "final_status": result.trace.final_status,
                "failure_type": result.trace.failure_type,
                "error_message": result.trace.error_message,
                "latency_ms": result.trace.latency_ms,
                "stages": [asdict(stage) for stage in result.trace.stages],
            },
            "metadata": dict(result.metadata),
        }

    @staticmethod
    def _serialize_document(document: IngestedDocument) -> dict[str, Any]:
        """把成功摄取的文档转换为报告摘要。"""

        raw_document = document.raw_document
        parsed_document = document.parsed_document
        return {
            "doc_id": parsed_document.doc_id,
            "version_id": parsed_document.version_id,
            "content_hash": parsed_document.content_hash,
            "title": parsed_document.title,
            "source_path": IngestionReportWriter._normalize_path(
                parsed_document.source_path
            ),
            "file_type": raw_document.file_type,
            "text_length": len(parsed_document.text),
            "block_count": len(parsed_document.blocks),
            "issue_count": len(parsed_document.parse_issues),
            "issues": [
                {
                    "code": issue.code,
                    "severity": issue.severity,
                    "message": issue.message,
                    "page": issue.page,
                    "section": issue.section,
                }
                for issue in parsed_document.parse_issues
            ],
        }

    @staticmethod
    def _normalize_path(path: Any) -> str:
        """把报告路径统一为 POSIX 风格。"""

        return Path(str(path)).as_posix() if path else ""
