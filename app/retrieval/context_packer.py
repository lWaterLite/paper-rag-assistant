"""上下文组织。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.core.models import Citation, RetrievedChunk


@dataclass(frozen=True)
class DroppedChunk:
    """没有进入最终上下文的 chunk。"""

    chunk_id: str
    reason: str
    detail: str


@dataclass(frozen=True)
class ContextCandidate:
    """准备进入上下文的候选片段。

    一个候选片段可能由多个相邻 chunk 合并而来。
    """

    text: str
    chunks: list[RetrievedChunk]


@dataclass(frozen=True)
class PackedContext:
    """准备交给生成模型的上下文。"""

    context_text: str
    citations: list[Citation]
    used_chunks: list[RetrievedChunk]
    dropped_chunks: list[DroppedChunk]


class ContextPacker(Protocol):
    """上下文组织器协议。"""

    def pack(self, chunks: list[RetrievedChunk]) -> PackedContext:
        """把检索结果组织成生成器可使用的上下文。"""


class SimpleContextPacker:
    """将检索结果转换成带 citation id 的上下文。"""

    def __init__(self, max_context_chars: int) -> None:
        self._max_context_chars = max_context_chars

    def pack(self, chunks: list[RetrievedChunk]) -> PackedContext:
        context_parts: list[str] = []
        citations: list[Citation] = []
        used_chunks: list[RetrievedChunk] = []
        dropped_chunks: list[DroppedChunk] = []
        current_length = 0

        unique_chunks = self._deduplicate_chunks(chunks, dropped_chunks)
        candidates = self._merge_adjacent_chunks(unique_chunks)

        for candidate_index, candidate in enumerate(candidates):
            citation_id = f"C{len(citations) + 1}"
            separator_length = 2 if context_parts else 0
            prefix = f"[{citation_id}] "
            remaining_chars = (
                self._max_context_chars - current_length - separator_length
            )
            candidate_text = self._fit_text_to_budget(
                candidate.text, len(prefix), remaining_chars
            )

            if candidate_text is None:
                for chunk in self._iter_candidate_chunks(candidates[candidate_index:]):
                    dropped_chunks.append(
                        DroppedChunk(
                            chunk_id=chunk.chunk_id,
                            reason="context_budget_exceeded",
                            detail="剩余上下文预算不足，无法放入该 chunk",
                        )
                    )
                break

            part = f"{prefix}{candidate_text}"
            context_parts.append(part)
            used_chunks.extend(candidate.chunks)
            current_length += separator_length + len(part)
            citations.append(
                self._build_citation(citation_id, candidate, candidate_text)
            )

        return PackedContext(
            context_text="\n\n".join(context_parts),
            citations=citations,
            used_chunks=used_chunks,
            dropped_chunks=dropped_chunks,
        )

    def _deduplicate_chunks(
        self,
        chunks: list[RetrievedChunk],
        dropped_chunks: list[DroppedChunk],
    ) -> list[RetrievedChunk]:
        """按 chunk_id 和文本内容去重。"""

        seen_chunk_ids: set[str] = set()
        seen_texts: set[str] = set()
        unique_chunks: list[RetrievedChunk] = []

        for chunk in chunks:
            normalized_text = " ".join(chunk.text.split()).lower()

            if chunk.chunk_id in seen_chunk_ids:
                dropped_chunks.append(
                    DroppedChunk(
                        chunk_id=chunk.chunk_id,
                        reason="duplicate_chunk_id",
                        detail="同一个 chunk_id 已经进入候选上下文",
                    )
                )
                continue

            if normalized_text in seen_texts:
                dropped_chunks.append(
                    DroppedChunk(
                        chunk_id=chunk.chunk_id,
                        reason="duplicate_content",
                        detail="文本内容与已有候选 chunk 重复",
                    )
                )
                continue

            seen_chunk_ids.add(chunk.chunk_id)
            seen_texts.add(normalized_text)
            unique_chunks.append(chunk)

        return unique_chunks

    def _merge_adjacent_chunks(
        self, chunks: list[RetrievedChunk]
    ) -> list[ContextCandidate]:
        """合并同一文档中的相邻 chunk。"""

        candidates: list[ContextCandidate] = []

        for chunk in chunks:
            if candidates and self._is_adjacent_chunk(candidates[-1].chunks[-1], chunk):
                previous = candidates[-1]
                candidates[-1] = ContextCandidate(
                    text=f"{previous.text}\n{chunk.text}",
                    chunks=[*previous.chunks, chunk],
                )
                continue

            candidates.append(ContextCandidate(text=chunk.text, chunks=[chunk]))

        return candidates

    def _is_adjacent_chunk(self, left: RetrievedChunk, right: RetrievedChunk) -> bool:
        """判断两个 chunk 是否来自同一文档的相邻位置。"""

        return (
            left.doc_id == right.doc_id
            and left.version_id == right.version_id
            and left.chunk_index + 1 == right.chunk_index
        )

    def _fit_text_to_budget(
        self, text: str, prefix_length: int, remaining_chars: int
    ) -> str | None:
        """把候选文本放入剩余上下文预算。

        如果候选文本过长，则截断。这里先做简单截断，后续接入真实 LLM 后可以替换成摘要。
        """

        if remaining_chars <= prefix_length:
            return None

        text_budget = remaining_chars - prefix_length
        if len(text) <= text_budget:
            return text

        if text_budget <= 3:
            return None

        return text[: text_budget - 3].rstrip() + "..."

    def _build_citation(
        self,
        citation_id: str,
        candidate: ContextCandidate,
        packed_text: str,
    ) -> Citation:
        """根据候选片段创建 citation。"""

        first_chunk = candidate.chunks[0]
        page_starts = [
            chunk.page_start
            for chunk in candidate.chunks
            if chunk.page_start is not None
        ]
        page_ends = [
            chunk.page_end for chunk in candidate.chunks if chunk.page_end is not None
        ]

        return Citation(
            citation_id=citation_id,
            chunk_id=first_chunk.chunk_id,
            doc_id=first_chunk.doc_id,
            version_id=first_chunk.version_id,
            title=first_chunk.title,
            source_path=first_chunk.source_path,
            snippet=packed_text[:180],
            page_start=min(page_starts) if page_starts else None,
            page_end=max(page_ends) if page_ends else None,
            section=first_chunk.section,
        )

    def _iter_candidate_chunks(
        self, candidates: list[ContextCandidate]
    ) -> list[RetrievedChunk]:
        """展开候选片段中的原始 chunk。"""

        return [chunk for candidate in candidates for chunk in candidate.chunks]
