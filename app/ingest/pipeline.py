"""文档摄取 pipeline。

真实工程中，批量摄取文档时不能因为单个坏文件就中断整个任务。
本模块负责把 loading、parsing、cleaning 串起来，并输出可观测的摄取报告。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.core.errors import AppError, ErrorCode
from app.core.models import ParsedDocument, RagTrace, RawDocument
from app.ingest.loaders import DocumentLoader
from app.ingest.parsers import ParserRegistry


@dataclass(frozen=True)
class IngestionFailure:
    """单个文件的摄取失败记录。"""

    source_path: str
    stage: str
    error_code: str
    error_message: str


@dataclass(frozen=True)
class IngestedDocument:
    """摄取成功的文档。"""

    raw_document: RawDocument
    parsed_document: ParsedDocument


@dataclass(frozen=True)
class IngestionResult:
    """一次目录摄取的结果。"""

    documents: list[IngestedDocument]
    failures: list[IngestionFailure]
    trace: RagTrace
    metadata: dict[str, int | str] = field(default_factory=dict)

    @property
    def raw_documents(self) -> list[RawDocument]:
        """返回成功摄取的原始文档。"""

        return [document.raw_document for document in self.documents]

    @property
    def parsed_documents(self) -> list[ParsedDocument]:
        """返回成功解析的文档。"""

        return [document.parsed_document for document in self.documents]


@dataclass(frozen=True)
class IngestionReportConfig:
    """摄取报告 writer 的运行时配置。"""

    output_dir: Path = Path("logs")

    @property
    def output_path(self) -> Path:
        """默认报告文件路径。"""

        return self.output_dir / "ingestion_report.json"


class IngestionReportWriter:
    """把摄取结果写成可观察、可排查的 JSON 报告。

    Writer 不参与文档加载和解析，只负责把已经完成的 IngestionResult 转成稳定的报告格式。
    这样 CLI、后台任务或测试都可以复用同一套报告生成逻辑。
    """

    def write(self, result: IngestionResult, output_path: Path) -> Path:
        """把摄取报告写入指定 JSON 文件，并返回实际写入路径。"""

        report = self.build_report(result)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
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
        """把成功摄取的文档摘要转换成报告字段。"""

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
        """把报告中的路径统一成 POSIX 风格，便于跨平台比较和阅读。"""

        return Path(str(path)).as_posix() if path else ""


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
