"""chunking 质量检查。

ReportWriter 负责描述 chunking 结果；QualityChecker 负责判断这些结果是否达标。
本模块不读取配置文件、不写文件、不创建目录，也不调用 chunker 或 embedding。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Literal

from app.core.models import DocumentChunk, ParsedDocument


IssueSeverity = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class ChunkingQualityConfig:
    """chunking 质量检查规则。

    这些配置描述“切分结果是否可接受”，而不是描述“如何切分”。
    """

    allow_empty_chunks: bool = False
    require_doc_id: bool = True
    require_source_path: bool = True
    min_avg_token_count: int | None = 1
    max_avg_token_count: int | None = 1200
    max_missing_pdf_page_ratio: float = 0.05
    max_missing_section_ratio: float = 0.5
    avg_token_issue_severity: IssueSeverity = "warning"
    missing_section_issue_severity: IssueSeverity = "warning"

    def __post_init__(self) -> None:
        """校验质量规则自身是否合法。"""

        if self.min_avg_token_count is not None and self.min_avg_token_count < 0:
            raise ValueError("min_avg_token_count 必须大于等于 0")
        if self.max_avg_token_count is not None and self.max_avg_token_count <= 0:
            raise ValueError("max_avg_token_count 必须大于 0")
        if (
            self.min_avg_token_count is not None
            and self.max_avg_token_count is not None
            and self.min_avg_token_count > self.max_avg_token_count
        ):
            raise ValueError("min_avg_token_count 必须小于等于 max_avg_token_count")
        _validate_ratio("max_missing_pdf_page_ratio", self.max_missing_pdf_page_ratio)
        _validate_ratio("max_missing_section_ratio", self.max_missing_section_ratio)


@dataclass(frozen=True)
class ChunkingQualityIssue:
    """一条 chunking 质量问题。"""

    code: str
    message: str
    severity: IssueSeverity
    value: int | float | str | None = None
    threshold: int | float | str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkingQualityCheckResult:
    """一次 chunking 质量检查结果。"""

    issues: list[ChunkingQualityIssue]
    checked_document_count: int
    checked_chunk_count: int

    @property
    def passed(self) -> bool:
        """只要不存在 error 级别问题，就认为质量检查通过。"""

        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def error_count(self) -> int:
        """error 级别问题数量。"""

        return len([issue for issue in self.issues if issue.severity == "error"])

    @property
    def warning_count(self) -> int:
        """warning 级别问题数量。"""

        return len([issue for issue in self.issues if issue.severity == "warning"])


class ChunkingQualityChecker:
    """检查 chunking 结果是否满足工程质量要求。"""

    def check(
        self,
        *,
        documents: list[ParsedDocument],
        chunks: list[DocumentChunk],
        config: ChunkingQualityConfig,
    ) -> ChunkingQualityCheckResult:
        """执行 chunking 质量检查。"""

        issues: list[ChunkingQualityIssue] = []
        issues.extend(self._check_empty_result(documents=documents, chunks=chunks))
        issues.extend(
            self._check_required_identity_fields(chunks=chunks, config=config)
        )
        issues.extend(self._check_empty_chunks(chunks=chunks, config=config))
        issues.extend(self._check_avg_token_count(chunks=chunks, config=config))
        issues.extend(self._check_pdf_page_ratio(chunks=chunks, config=config))
        issues.extend(self._check_section_ratio(chunks=chunks, config=config))

        return ChunkingQualityCheckResult(
            issues=issues,
            checked_document_count=len(documents),
            checked_chunk_count=len(chunks),
        )

    @staticmethod
    def _check_empty_result(
        *,
        documents: list[ParsedDocument],
        chunks: list[DocumentChunk],
    ) -> list[ChunkingQualityIssue]:
        """检查有输入文档但没有产生任何 chunk 的情况。"""

        if documents and not chunks:
            return [
                ChunkingQualityIssue(
                    code="no_chunks_created",
                    message="存在已解析文档，但 chunking 阶段没有产生任何 chunk",
                    severity="error",
                    value=0,
                    threshold="> 0",
                )
            ]
        return []

    @staticmethod
    def _check_required_identity_fields(
        *,
        chunks: list[DocumentChunk],
        config: ChunkingQualityConfig,
    ) -> list[ChunkingQualityIssue]:
        """检查检索和引用所需的基础身份字段。"""

        issues: list[ChunkingQualityIssue] = []

        if config.require_doc_id:
            missing_doc_id_count = len([chunk for chunk in chunks if not chunk.doc_id])
            if missing_doc_id_count:
                issues.append(
                    ChunkingQualityIssue(
                        code="missing_doc_id",
                        message="部分 chunk 缺失 doc_id，后续无法稳定关联原始文档",
                        severity="error",
                        value=missing_doc_id_count,
                        threshold=0,
                    )
                )

        if config.require_source_path:
            missing_source_path_count = len(
                [chunk for chunk in chunks if not chunk.source_path]
            )
            if missing_source_path_count:
                issues.append(
                    ChunkingQualityIssue(
                        code="missing_source_path",
                        message="部分 chunk 缺失 source_path，后续引用和排障会失去来源路径",
                        severity="error",
                        value=missing_source_path_count,
                        threshold=0,
                    )
                )

        return issues

    @staticmethod
    def _check_empty_chunks(
        *,
        chunks: list[DocumentChunk],
        config: ChunkingQualityConfig,
    ) -> list[ChunkingQualityIssue]:
        """检查空 chunk。"""

        if config.allow_empty_chunks:
            return []

        empty_chunk_count = len([chunk for chunk in chunks if not chunk.text.strip()])
        if not empty_chunk_count:
            return []

        return [
            ChunkingQualityIssue(
                code="empty_chunk_found",
                message="chunking 结果中存在空 chunk",
                severity="error",
                value=empty_chunk_count,
                threshold=0,
            )
        ]

    @staticmethod
    def _check_avg_token_count(
        *,
        chunks: list[DocumentChunk],
        config: ChunkingQualityConfig,
    ) -> list[ChunkingQualityIssue]:
        """检查平均 token 数是否落在期望区间。"""

        if not chunks:
            return []

        avg_token_count = mean(chunk.token_count for chunk in chunks)
        issues: list[ChunkingQualityIssue] = []

        if (
            config.min_avg_token_count is not None
            and avg_token_count < config.min_avg_token_count
        ):
            issues.append(
                ChunkingQualityIssue(
                    code="avg_token_count_too_low",
                    message="平均 token 数过低，chunk 可能切得过碎",
                    severity=config.avg_token_issue_severity,
                    value=round(avg_token_count, 2),
                    threshold=config.min_avg_token_count,
                )
            )

        if (
            config.max_avg_token_count is not None
            and avg_token_count > config.max_avg_token_count
        ):
            issues.append(
                ChunkingQualityIssue(
                    code="avg_token_count_too_high",
                    message="平均 token 数过高，chunk 可能过长并影响召回精度或上下文预算",
                    severity=config.avg_token_issue_severity,
                    value=round(avg_token_count, 2),
                    threshold=config.max_avg_token_count,
                )
            )

        return issues

    @staticmethod
    def _check_pdf_page_ratio(
        *,
        chunks: list[DocumentChunk],
        config: ChunkingQualityConfig,
    ) -> list[ChunkingQualityIssue]:
        """检查 PDF chunk 的页码缺失比例。"""

        pdf_chunks = [chunk for chunk in chunks if _looks_like_pdf_chunk(chunk)]
        if not pdf_chunks:
            return []

        missing_page_count = len(
            [chunk for chunk in pdf_chunks if chunk.page_start is None]
        )
        missing_page_ratio = missing_page_count / len(pdf_chunks)
        if missing_page_ratio <= config.max_missing_pdf_page_ratio:
            return []

        return [
            ChunkingQualityIssue(
                code="missing_pdf_page_ratio_too_high",
                message="PDF chunk 缺失页码比例过高，可能影响论文引用可追溯性",
                severity="error",
                value=round(missing_page_ratio, 4),
                threshold=config.max_missing_pdf_page_ratio,
                metadata={
                    "pdf_chunk_count": len(pdf_chunks),
                    "missing_page_count": missing_page_count,
                },
            )
        ]

    @staticmethod
    def _check_section_ratio(
        *,
        chunks: list[DocumentChunk],
        config: ChunkingQualityConfig,
    ) -> list[ChunkingQualityIssue]:
        """检查 section 缺失比例。"""

        if not chunks:
            return []

        missing_section_count = len([chunk for chunk in chunks if not chunk.section])
        missing_section_ratio = missing_section_count / len(chunks)
        if missing_section_ratio <= config.max_missing_section_ratio:
            return []

        return [
            ChunkingQualityIssue(
                code="missing_section_ratio_too_high",
                message="chunk 缺失 section 的比例过高，可能影响按章节检索、引用和质量分析",
                severity=config.missing_section_issue_severity,
                value=round(missing_section_ratio, 4),
                threshold=config.max_missing_section_ratio,
                metadata={
                    "chunk_count": len(chunks),
                    "missing_section_count": missing_section_count,
                },
            )
        ]


def _looks_like_pdf_chunk(chunk: DocumentChunk) -> bool:
    """判断 chunk 是否来自 PDF 文档。"""

    return str(
        chunk.metadata.get("suffix") or ""
    ).lower() == ".pdf" or chunk.source_path.lower().endswith(".pdf")


def _validate_ratio(name: str, value: float) -> None:
    """校验比例配置必须处于 [0, 1] 区间。"""

    if not 0 <= value <= 1:
        raise ValueError(f"{name} 必须处于 0 到 1 之间")
