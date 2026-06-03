"""上下文组织。"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.models import Citation, RetrievedChunk


@dataclass(frozen=True)
class PackedContext:
    """准备交给生成模型的上下文。"""

    context_text: str
    citations: list[Citation]
    used_chunks: list[RetrievedChunk]


class SimpleContextPacker:
    """将检索结果转换成带 citation id 的上下文。"""

    def __init__(self, max_context_chars: int) -> None:
        self._max_context_chars = max_context_chars

    def pack(self, chunks: list[RetrievedChunk]) -> PackedContext:
        context_parts: list[str] = []
        citations: list[Citation] = []
        used_chunks: list[RetrievedChunk] = []
        current_length = 0

        for index, chunk in enumerate(chunks, start=1):
            citation_id = f"C{index}"
            part = f"[{citation_id}] {chunk.text}"
            if current_length + len(part) > self._max_context_chars:
                break

            context_parts.append(part)
            used_chunks.append(chunk)
            current_length += len(part)
            citations.append(
                Citation(
                    citation_id=citation_id,
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    version_id=chunk.version_id,
                    title=chunk.title,
                    source_path=chunk.source_path,
                    snippet=chunk.text[:180],
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section=chunk.section,
                )
            )

        return PackedContext(
            context_text="\n\n".join(context_parts),
            citations=citations,
            used_chunks=used_chunks,
        )

    # TODO 练习 10：
    # 当前 context packing 只按检索顺序截断。
    # 请你尝试改进：
    # 1. 同一文档相邻 chunk 合并。
    # 2. 重复内容去重。
    # 3. 当一个 chunk 超长时做摘要或截断。
    # 4. 在返回结果中说明哪些 chunk 被丢弃以及原因。
