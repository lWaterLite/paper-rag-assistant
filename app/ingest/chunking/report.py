"""chunking 质量报告。

报告组件只负责把 chunking 结果转换为可序列化结构并写入文件。
目录创建和流程编排由索引构建流程负责。
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from app.core.models import DocumentChunk, ParsedDocument
from app.ingest.chunking.strategies import ChunkerConfig


@dataclass(frozen=True)
class ChunkingReportConfig:
    """chunking 报告 writer 的运行时配置。"""

    output_dir: Path = Path("logs")

    @property
    def output_path(self) -> Path:
        """默认报告文件路径。"""

        return self.output_dir / "chunking_report.json"


class ChunkingReportWriter:
    """把 chunking 质量信息写成 JSON 报告。"""

    def write(
        self,
        *,
        documents: list[ParsedDocument],
        chunks: list[DocumentChunk],
        config: ChunkerConfig,
        output_path: Path,
    ) -> Path:
        """写入 chunking 报告，并返回报告路径。"""

        report = self.build_report(documents=documents, chunks=chunks, config=config)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return output_path

    @staticmethod
    def build_report(
        *,
        documents: list[ParsedDocument],
        chunks: list[DocumentChunk],
        config: ChunkerConfig,
    ) -> dict[str, Any]:
        """构建可 JSON 序列化的 chunking 质量报告。"""

        token_counts = [chunk.token_count for chunk in chunks]
        empty_chunks = [chunk for chunk in chunks if not chunk.text.strip()]
        missing_doc_id = [chunk for chunk in chunks if not chunk.doc_id]
        missing_source_path = [chunk for chunk in chunks if not chunk.source_path]
        missing_page = [chunk for chunk in chunks if chunk.page_start is None and _looks_like_pdf_chunk(chunk)]
        missing_section = [chunk for chunk in chunks if not chunk.section]

        return {
            "chunker": _infer_chunker_name(chunks),
            "strategy": config.strategy,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "tokenizer": config.tokenizer,
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "avg_token_count": round(mean(token_counts), 2) if token_counts else 0,
            "min_token_count": min(token_counts) if token_counts else 0,
            "max_token_count": max(token_counts) if token_counts else 0,
            "empty_chunk_count": len(empty_chunks),
            "missing_doc_id_count": len(missing_doc_id),
            "missing_source_path_count": len(missing_source_path),
            "missing_page_count": len(missing_page),
            "missing_section_count": len(missing_section),
            "documents": _build_document_summaries(documents, chunks),
        }


def _build_document_summaries(
    documents: list[ParsedDocument],
    chunks: list[DocumentChunk],
) -> list[dict[str, Any]]:
    """按文档聚合 chunk 统计。"""

    chunks_by_doc_id: dict[str, list[DocumentChunk]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_doc_id[chunk.doc_id].append(chunk)

    summaries: list[dict[str, Any]] = []
    for document in documents:
        document_chunks = chunks_by_doc_id.get(document.doc_id, [])
        token_counts = [chunk.token_count for chunk in document_chunks]
        summaries.append(
            {
                "doc_id": document.doc_id,
                "version_id": document.version_id,
                "title": document.title,
                "source_path": Path(document.source_path).as_posix(),
                "chunk_count": len(document_chunks),
                "avg_token_count": round(mean(token_counts), 2) if token_counts else 0,
                "max_token_count": max(token_counts) if token_counts else 0,
                "missing_page_count": len(
                    [chunk for chunk in document_chunks if chunk.page_start is None and _looks_like_pdf_chunk(chunk)]
                ),
                "missing_section_count": len([chunk for chunk in document_chunks if not chunk.section]),
            }
        )

    return summaries


def _looks_like_pdf_chunk(chunk: DocumentChunk) -> bool:
    """判断 chunk 是否来自 PDF 文档。"""

    return str(chunk.metadata.get("suffix") or "").lower() == ".pdf" or chunk.source_path.lower().endswith(".pdf")


def _infer_chunker_name(chunks: list[DocumentChunk]) -> str:
    """从 chunk metadata 中推断 chunker 名称。"""

    if not chunks:
        return "unknown"
    return str(chunks[0].metadata.get("chunker", "unknown"))
