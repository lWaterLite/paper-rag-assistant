"""索引构建报告。

IndexBuilder 负责流程编排，ReportWriter 负责把构建结果转换成稳定 JSON。
这样 CLI、测试和后续 API 都能复用同一套报告结构。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.indexing.pipeline.types import IndexBuildResult


class IndexBuildReportWriter:
    """把索引构建结果写成 JSON 报告。"""

    def write(self, result: IndexBuildResult, output_path: Path) -> Path:
        """写入索引构建报告，并返回报告路径。"""

        output_path.write_text(
            json.dumps(
                self.build_report(result), ensure_ascii=False, indent=2, default=str
            ),
            encoding="utf-8",
        )
        return output_path

    @staticmethod
    def build_report(result: IndexBuildResult) -> dict[str, Any]:
        """构建可 JSON 序列化的索引构建报告。"""

        return {
            "index_id": result.manifest.index_id,
            "status": result.trace.final_status,
            "document_count": result.document_count,
            "chunk_count": result.chunk_count,
            "vector_count": result.vector_count,
            "embedding_cache_hits": result.embedding_cache_hits,
            "embedding_cache_misses": result.embedding_cache_misses,
            "skipped_existing_chunks": result.skipped_existing_chunks,
            "empty_chunk_count": result.empty_chunk_count,
            "manifest_path": _path_to_string(result.manifest_path),
            "ingestion_report_path": _path_to_string(result.ingestion_report_path),
            "chunking_report_path": _path_to_string(result.chunking_report_path),
            "manifest": result.manifest.to_dict(),
            "trace": {
                "trace_id": result.trace.trace_id,
                "final_status": result.trace.final_status,
                "failure_type": result.trace.failure_type,
                "error_message": result.trace.error_message,
                "latency_ms": result.trace.latency_ms,
                "stages": [asdict(stage) for stage in result.trace.stages],
            },
        }


def _path_to_string(path: Path | None) -> str | None:
    """把路径转成跨平台更稳定的 POSIX 字符串。"""

    return path.as_posix() if path is not None else None
