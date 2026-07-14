"""文档摄取 pipeline。

真实工程中，批量摄取文档时不能因为单个坏文件就中断整个任务。
本模块负责把 loading、parsing、cleaning 串起来，并输出可观测的摄取报告。
"""

from __future__ import annotations

import time
from pathlib import Path

from app.core.errors import AppError, ErrorCode
from app.core.tracing import RagTrace
from app.ingest.loading.local import DocumentLoader
from app.ingest.models import ParsedDocument, RawDocument
from app.ingest.parsing.parsers import ParserRegistry
from app.ingest.pipeline_types import IngestedDocument, IngestionFailure, IngestionResult


class IngestionPipeline:
    """本地文档摄取 pipeline。"""

    def __init__(
        self,
        loader: DocumentLoader,
        parser_registry: ParserRegistry,
    ) -> None:
        self._loader = loader
        self._parser_registry = parser_registry

    def ingest_directory(self, source_dir: Path) -> IngestionResult:
        """摄取目录中的所有支持文件。"""

        if not source_dir.exists():
            raise AppError(
                ErrorCode.DOCUMENT_LOAD_FAILED, f"文档目录不存在：{source_dir}"
            )

        if not source_dir.is_dir():
            raise AppError(
                ErrorCode.DOCUMENT_LOAD_FAILED, f"文档来源不是目录：{source_dir}"
            )

        trace = RagTrace()
        documents: list[IngestedDocument] = []
        failures: list[IngestionFailure] = []

        started = time.perf_counter()
        paths = list(self._loader.iter_supported_files(source_dir))
        trace.record_stage(
            "discovering", "success", started, {"candidate_files": len(paths)}
        )

        for path in paths:
            raw_document = self._load_file(path, failures)
            if raw_document is None:
                continue

            parsed_document = self._parse_document(raw_document, failures)
            if parsed_document is None:
                continue

            documents.append(
                IngestedDocument(
                    raw_document=raw_document, parsed_document=parsed_document
                )
            )

        trace.mark_success()
        return IngestionResult(
            documents=documents,
            failures=failures,
            trace=trace,
            metadata={
                "source_dir": str(source_dir),
                "candidate_files": len(paths),
                "succeeded": len(documents),
                "failed": len(failures),
            },
        )

    def _load_file(
        self, path: Path, failures: list[IngestionFailure]
    ) -> RawDocument | None:
        """加载单个文件，失败时记录并跳过。"""

        try:
            return self._loader.load_file(path)
        except AppError as exc:
            failures.append(
                IngestionFailure(
                    source_path=str(path),
                    stage="loading",
                    error_code=exc.code.value,
                    error_message=exc.message,
                )
            )
            return None

    def _parse_document(
        self,
        raw_document: RawDocument,
        failures: list[IngestionFailure],
    ) -> ParsedDocument | None:
        """解析单个文件，失败时记录并跳过。"""

        try:
            return self._parser_registry.parse(raw_document)
        except AppError as exc:
            failures.append(
                IngestionFailure(
                    source_path=raw_document.source_path,
                    stage="parsing",
                    error_code=exc.code.value,
                    error_message=exc.message,
                )
            )
            return None
