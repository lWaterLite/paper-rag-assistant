"""RAG 系统中的核心数据模型。

子模块 1 的重点不是复杂算法，而是先把数据在 pipeline 中如何流动讲清楚。
这些 dataclass 后续可以替换成 Pydantic model，用于 API schema 和字段校验。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal


StageStatus = Literal["success", "error"]


def new_id(prefix: str) -> str:
    """生成带前缀的短 ID，方便阅读 trace 和调试输出。"""

    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True)
class RawDocument:
    """原始文档。

    这是 loading 阶段的输出，表示系统刚刚从磁盘或其他来源读到的文档。
    """

    doc_id: str
    source_path: str
    file_type: str
    content_hash: str
    version_id: str
    raw_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedDocument:
    """解析后的文档。

    这是 parsing/cleaning 阶段的输出，后续会被切分为 chunk。
    """

    doc_id: str
    content_hash: str
    version_id: str
    title: str
    text: str
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentChunk:
    """文档切分后的片段。

    chunk 是 RAG 检索的基本单位。后续 embedding、检索、引用都围绕 chunk 展开。
    """

    chunk_id: str
    doc_id: str
    content_hash: str
    version_id: str
    text: str
    source_path: str
    chunk_index: int
    token_count: int
    title: str | None = None
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    """检索结果。

    与 DocumentChunk 相比，它多了 score、rank、retriever 等查询时产生的信息。
    """

    chunk_id: str
    doc_id: str
    content_hash: str
    version_id: str
    text: str
    score: float
    rank: int
    retriever: str
    source_path: str
    title: str | None = None
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Citation:
    """回答中的引用来源。"""

    citation_id: str
    chunk_id: str
    doc_id: str
    version_id: str
    title: str | None
    source_path: str
    snippet: str
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None


@dataclass(frozen=True)
class RagAnswer:
    """一次 RAG 问答的最终结构化结果。"""

    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    trace_id: str
    latency_ms: float


@dataclass(frozen=True)
class PipelineStageRun:
    """一次 pipeline 阶段运行记录。"""

    stage: str
    status: StageStatus
    latency_ms: float
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class RagTrace:
    """一次请求的完整追踪信息。"""

    trace_id: str = field(default_factory=lambda: new_id("trace"))
    started_at: float = field(default_factory=time.perf_counter)
    stages: list[PipelineStageRun] = field(default_factory=list)

    def record_stage(
        self,
        stage: str,
        status: StageStatus,
        started_at: float,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """记录一个阶段的耗时和摘要信息。"""

        latency_ms = (time.perf_counter() - started_at) * 1000
        self.stages.append(
            PipelineStageRun(
                stage=stage,
                status=status,
                latency_ms=round(latency_ms, 2),
                detail=detail or {},
            )
        )

    @property
    def latency_ms(self) -> float:
        """整个请求从开始到当前的耗时。"""

        return round((time.perf_counter() - self.started_at) * 1000, 2)
