"""Token-aware 上下文组织与证据来源追溯。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from app.core.models import Citation, RetrievedChunk
from app.retrieval.token_estimators import TokenEstimator


@dataclass(frozen=True, slots=True)
class ContextPackerConfig:
    """ContextPacker 实际使用的 token 预算与选择策略。"""

    model_context_window: int = 4096
    max_context_tokens: int = 1800
    reserved_prompt_tokens: int = 200
    reserved_output_tokens: int = 512
    safety_margin_tokens: int = 64
    max_chunks_per_document: int = 2

    def __post_init__(self) -> None:
        numeric_values = {
            "model_context_window": self.model_context_window,
            "max_context_tokens": self.max_context_tokens,
            "reserved_prompt_tokens": self.reserved_prompt_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "safety_margin_tokens": self.safety_margin_tokens,
            "max_chunks_per_document": self.max_chunks_per_document,
        }
        invalid_names = [name for name, value in numeric_values.items() if value <= 0]
        if invalid_names:
            raise ValueError(f"context packing 配置必须大于 0：{', '.join(invalid_names)}")
        if self.max_context_tokens > self.model_context_window:
            raise ValueError("max_context_tokens 不能大于 model_context_window")


@dataclass(frozen=True, slots=True)
class ContextPackRequest:
    """ContextPacker 的结构化输入。"""

    query: str
    chunks: Sequence[RetrievedChunk]


@dataclass(frozen=True, slots=True)
class DroppedChunk:
    """没有进入最终上下文的 chunk 及其原因。"""

    chunk_id: str
    reason: str
    detail: str


@dataclass(frozen=True, slots=True)
class ContextCandidate:
    """可进入上下文的候选段，可能由相邻 chunk 合并而成。"""

    text: str
    chunks: tuple[RetrievedChunk, ...]


@dataclass(frozen=True, slots=True)
class ContextSegment:
    """最终上下文中的一个可追溯证据段。"""

    citation_id: str
    text: str
    source_chunk_ids: tuple[str, ...]
    source_doc_id: str
    page_start: int | None
    page_end: int | None
    sections: tuple[str, ...]
    token_count: int
    is_truncated: bool


@dataclass(frozen=True, slots=True)
class ContextTokenUsage:
    """一次 context packing 的 token 预算明细。"""

    estimator: str
    question_tokens: int
    reserved_prompt_tokens: int
    reserved_output_tokens: int
    safety_margin_tokens: int
    available_context_tokens: int
    used_context_tokens: int

    @property
    def remaining_context_tokens(self) -> int:
        """返回资料上下文可继续使用的 token 余额。"""

        return self.available_context_tokens - self.used_context_tokens


@dataclass(frozen=True, slots=True)
class PackedContext:
    """准备交给生成模型的上下文与完整来源映射。"""

    context_text: str
    citations: list[Citation]
    used_chunks: list[RetrievedChunk]
    dropped_chunks: list[DroppedChunk]
    segments: list[ContextSegment]
    token_usage: ContextTokenUsage


class ContextPacker(Protocol):
    """根据 query、候选结果和 token 预算组织上下文。"""

    def pack(self, request: ContextPackRequest) -> PackedContext:
        """把检索候选组织为可引用、可追溯的上下文。"""


class TokenAwareContextPacker:
    """按 token 预算选择证据段，并保留完整 chunk 来源。"""

    def __init__(self, *, config: ContextPackerConfig, token_estimator: TokenEstimator) -> None:
        self._config = config
        self._token_estimator = token_estimator

    def pack(self, request: ContextPackRequest) -> PackedContext:
        """在模型窗口约束内组装上下文。"""

        question_tokens = self._token_estimator.count_text(request.query)
        available_context_tokens = self._resolve_available_context_tokens(question_tokens)
        dropped_chunks: list[DroppedChunk] = []
        unique_chunks = self._deduplicate_chunks(request.chunks, dropped_chunks)
        quota_chunks = self._apply_document_quota(unique_chunks, dropped_chunks)
        candidates = self._merge_adjacent_chunks(quota_chunks)

        context_parts: list[str] = []
        citations: list[Citation] = []
        used_chunks: list[RetrievedChunk] = []
        segments: list[ContextSegment] = []
        used_context_tokens = 0

        for candidate_index, candidate in enumerate(candidates):
            citation_id = f"C{len(citations) + 1}"
            prefix = f"[{citation_id}] "
            separator = "\n\n" if context_parts else ""
            remaining_tokens = (
                available_context_tokens
                - used_context_tokens
                - self._token_estimator.count_text(separator)
            )
            fitted_text, is_truncated = self._fit_text_to_budget(
                candidate.text,
                prefix=prefix,
                token_budget=remaining_tokens,
            )
            if fitted_text is None:
                self._drop_remaining_candidates(
                    candidates[candidate_index:],
                    dropped_chunks,
                    reason="context_token_budget_exceeded",
                    detail="剩余 token 预算不足，无法放入该上下文段",
                )
                break

            part = f"{prefix}{fitted_text}"
            part_token_count = self._token_estimator.count_text(f"{separator}{part}")
            context_parts.append(part)
            used_context_tokens += part_token_count
            used_chunks.extend(candidate.chunks)
            segment = self._build_segment(
                citation_id,
                candidate,
                fitted_text,
                token_count=part_token_count,
                is_truncated=is_truncated,
            )
            segments.append(segment)
            citations.append(self._build_citation(segment, candidate))

        return PackedContext(
            context_text="\n\n".join(context_parts),
            citations=citations,
            used_chunks=used_chunks,
            dropped_chunks=dropped_chunks,
            segments=segments,
            token_usage=ContextTokenUsage(
                estimator=self._token_estimator.name,
                question_tokens=question_tokens,
                reserved_prompt_tokens=self._config.reserved_prompt_tokens,
                reserved_output_tokens=self._config.reserved_output_tokens,
                safety_margin_tokens=self._config.safety_margin_tokens,
                available_context_tokens=available_context_tokens,
                used_context_tokens=used_context_tokens,
            ),
        )

    def _resolve_available_context_tokens(self, question_tokens: int) -> int:
        """从窗口、问题和预留空间推导资料可用预算。"""

        window_budget = (
            self._config.model_context_window
            - question_tokens
            - self._config.reserved_prompt_tokens
            - self._config.reserved_output_tokens
            - self._config.safety_margin_tokens
        )
        return max(0, min(self._config.max_context_tokens, window_budget))

    @staticmethod
    def _deduplicate_chunks(
        chunks: Sequence[RetrievedChunk],
        dropped_chunks: list[DroppedChunk],
    ) -> list[RetrievedChunk]:
        """按 chunk id 与规范化文本去重。"""

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
                        detail="同一个 chunk_id 已经进入上下文候选",
                    )
                )
                continue
            if normalized_text in seen_texts:
                dropped_chunks.append(
                    DroppedChunk(
                        chunk_id=chunk.chunk_id,
                        reason="duplicate_content",
                        detail="文本内容与已有上下文候选重复",
                    )
                )
                continue
            seen_chunk_ids.add(chunk.chunk_id)
            seen_texts.add(normalized_text)
            unique_chunks.append(chunk)
        return unique_chunks

    def _apply_document_quota(
        self,
        chunks: Sequence[RetrievedChunk],
        dropped_chunks: list[DroppedChunk],
    ) -> list[RetrievedChunk]:
        """限制单篇文档占用过多上下文候选。"""

        counts: dict[tuple[str, str], int] = {}
        selected: list[RetrievedChunk] = []
        for chunk in chunks:
            key = (chunk.doc_id, chunk.version_id)
            current_count = counts.get(key, 0)
            if current_count >= self._config.max_chunks_per_document:
                dropped_chunks.append(
                    DroppedChunk(
                        chunk_id=chunk.chunk_id,
                        reason="document_chunk_quota_exceeded",
                        detail=(
                            "同一文档已达到上下文候选上限："
                            f"{self._config.max_chunks_per_document}"
                        ),
                    )
                )
                continue
            counts[key] = current_count + 1
            selected.append(chunk)
        return selected

    @staticmethod
    def _merge_adjacent_chunks(
        chunks: Sequence[RetrievedChunk],
    ) -> list[ContextCandidate]:
        """在检索排序顺序中合并同文档相邻 chunk。"""

        candidates: list[ContextCandidate] = []
        for chunk in chunks:
            if candidates and _is_adjacent(candidates[-1].chunks[-1], chunk):
                previous = candidates[-1]
                candidates[-1] = ContextCandidate(
                    text=f"{previous.text}\n{chunk.text}",
                    chunks=(*previous.chunks, chunk),
                )
            else:
                candidates.append(ContextCandidate(text=chunk.text, chunks=(chunk,)))
        return candidates

    def _fit_text_to_budget(
        self,
        text: str,
        *,
        prefix: str,
        token_budget: int,
    ) -> tuple[str | None, bool]:
        """完整放入或以可追溯的文本前缀截断到 token 预算内。"""

        if token_budget <= self._token_estimator.count_text(prefix):
            return None, False
        if self._token_estimator.count_text(f"{prefix}{text}") <= token_budget:
            return text, False

        suffix = "..."
        low, high = 0, len(text)
        best_text = ""
        while low <= high:
            middle = (low + high) // 2
            candidate = text[:middle].rstrip() + suffix
            if self._token_estimator.count_text(f"{prefix}{candidate}") <= token_budget:
                best_text = candidate
                low = middle + 1
            else:
                high = middle - 1
        if not best_text or best_text == suffix:
            return None, False
        return best_text, True

    @staticmethod
    def _drop_remaining_candidates(
        candidates: Sequence[ContextCandidate],
        dropped_chunks: list[DroppedChunk],
        *,
        reason: str,
        detail: str,
    ) -> None:
        """记录由于同一预算原因未进入上下文的剩余候选。"""

        for candidate in candidates:
            for chunk in candidate.chunks:
                dropped_chunks.append(
                    DroppedChunk(chunk_id=chunk.chunk_id, reason=reason, detail=detail)
                )

    def _build_segment(
        self,
        citation_id: str,
        candidate: ContextCandidate,
        text: str,
        *,
        token_count: int,
        is_truncated: bool,
    ) -> ContextSegment:
        """构建记录完整来源的上下文段。"""

        page_starts = [
            chunk.page_start for chunk in candidate.chunks if chunk.page_start is not None
        ]
        page_ends = [
            chunk.page_end for chunk in candidate.chunks if chunk.page_end is not None
        ]
        sections = tuple(
            dict.fromkeys(
                chunk.section for chunk in candidate.chunks if chunk.section is not None
            )
        )
        return ContextSegment(
            citation_id=citation_id,
            text=text,
            source_chunk_ids=tuple(chunk.chunk_id for chunk in candidate.chunks),
            source_doc_id=candidate.chunks[0].doc_id,
            page_start=min(page_starts) if page_starts else None,
            page_end=max(page_ends) if page_ends else None,
            sections=sections,
            token_count=token_count,
            is_truncated=is_truncated,
        )

    @staticmethod
    def _build_citation(segment: ContextSegment, candidate: ContextCandidate) -> Citation:
        """构建与段级 provenance 对应的兼容 citation。"""

        first_chunk = candidate.chunks[0]
        return Citation(
            citation_id=segment.citation_id,
            chunk_id=first_chunk.chunk_id,
            doc_id=segment.source_doc_id,
            version_id=first_chunk.version_id,
            title=first_chunk.title,
            source_path=first_chunk.source_path,
            snippet=segment.text[:180],
            page_start=segment.page_start,
            page_end=segment.page_end,
            section=first_chunk.section,
        )


def _is_adjacent(left: RetrievedChunk, right: RetrievedChunk) -> bool:
    """判断两个 chunk 是否来自同一文档版本的相邻位置。"""

    return (
        left.doc_id == right.doc_id
        and left.version_id == right.version_id
        and left.chunk_index + 1 == right.chunk_index
    )
