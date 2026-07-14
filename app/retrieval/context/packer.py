"""Token-aware 上下文组织与证据来源追溯。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from app.core.models import Citation, RetrievedChunk
from app.retrieval.context.evidence_transformers.models import EvidenceCandidate
from app.retrieval.context.token_estimators import TokenEstimator


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
    candidates: Sequence[EvidenceCandidate]


@dataclass(frozen=True, slots=True)
class DroppedChunk:
    """没有进入最终上下文的来源 chunk 及其原因。"""

    chunk_id: str
    reason: str
    detail: str


@dataclass(frozen=True, slots=True)
class ContextCandidate:
    """可进入上下文的候选段，可能由多个候选证据合并而成。"""

    text: str
    evidence: tuple[EvidenceCandidate, ...]

    @property
    def primary_chunk(self) -> RetrievedChunk:
        """返回用于相邻判断、文档配额与 Citation 的首个来源。"""

        return self.evidence[0].primary_chunk

    @property
    def source_chunks(self) -> tuple[RetrievedChunk, ...]:
        """返回候选段的去重来源 chunk。"""

        chunks_by_identity: dict[tuple[str, str], RetrievedChunk] = {}
        for item in self.evidence:
            for chunk in item.source_chunks:
                key = (chunk.chunk_id, chunk.version_id)
                chunks_by_identity.setdefault(key, chunk)
        return tuple(chunks_by_identity.values())


@dataclass(frozen=True, slots=True)
class ContextSourceRange:
    """最终上下文段在原始 chunk 中的来源范围。"""

    chunk_id: str
    version_id: str
    char_start: int
    char_end: int


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
    source_ranges: tuple[ContextSourceRange, ...]


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
    """根据 query、候选证据和 token 预算组织上下文。"""

    def pack(self, request: ContextPackRequest) -> PackedContext:
        """把候选证据组织为可引用、可追溯的上下文。"""


class TokenAwareContextPacker:
    """按 token 预算选择证据段，并保留完整的原始 chunk 来源。"""

    def __init__(self, *, config: ContextPackerConfig, token_estimator: TokenEstimator) -> None:
        self._config = config
        self._token_estimator = token_estimator

    def pack(self, request: ContextPackRequest) -> PackedContext:
        """在模型窗口约束内组装上下文。"""

        question_tokens = self._token_estimator.count_text(request.query)
        available_context_tokens = self._resolve_available_context_tokens(question_tokens)
        dropped_chunks: list[DroppedChunk] = []
        unique_evidence = self._deduplicate_evidence(
            request.candidates,
            dropped_chunks,
        )
        quota_evidence = self._apply_document_quota(unique_evidence, dropped_chunks)
        candidates = self._merge_adjacent_evidence(quota_evidence)

        context_parts: list[str] = []
        citations: list[Citation] = []
        used_chunks: list[RetrievedChunk] = []
        used_chunk_identities: set[tuple[str, str]] = set()
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
            self._append_used_chunks(
                candidate,
                used_chunks,
                used_chunk_identities,
            )
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
    def _deduplicate_evidence(
        evidence: Sequence[EvidenceCandidate],
        dropped_chunks: list[DroppedChunk],
    ) -> list[EvidenceCandidate]:
        """按 evidence identity 与规范化文本去重。"""

        seen_evidence_ids: set[str] = set()
        seen_texts: set[str] = set()
        unique_evidence: list[EvidenceCandidate] = []
        for item in evidence:
            normalized_text = " ".join(item.text.split()).lower()
            if item.evidence_id in seen_evidence_ids:
                _append_dropped_evidence(
                    item,
                    dropped_chunks,
                    reason="duplicate_evidence_id",
                    detail="同一个 evidence_id 已经进入上下文候选",
                )
                continue
            if normalized_text in seen_texts:
                _append_dropped_evidence(
                    item,
                    dropped_chunks,
                    reason="duplicate_content",
                    detail="文本内容与已有上下文候选重复",
                )
                continue
            seen_evidence_ids.add(item.evidence_id)
            seen_texts.add(normalized_text)
            unique_evidence.append(item)
        return unique_evidence

    def _apply_document_quota(
        self,
        evidence: Sequence[EvidenceCandidate],
        dropped_chunks: list[DroppedChunk],
    ) -> list[EvidenceCandidate]:
        """限制单篇文档占用过多上下文候选。"""

        counts: dict[tuple[str, str], int] = {}
        selected: list[EvidenceCandidate] = []
        for item in evidence:
            primary_chunk = item.primary_chunk
            key = (primary_chunk.doc_id, primary_chunk.version_id)
            current_count = counts.get(key, 0)
            if current_count >= self._config.max_chunks_per_document:
                _append_dropped_evidence(
                    item,
                    dropped_chunks,
                    reason="document_chunk_quota_exceeded",
                    detail=(
                        "同一文档已达到上下文候选上限："
                        f"{self._config.max_chunks_per_document}"
                    ),
                )
                continue
            counts[key] = current_count + 1
            selected.append(item)
        return selected

    @staticmethod
    def _merge_adjacent_evidence(
        evidence: Sequence[EvidenceCandidate],
    ) -> list[ContextCandidate]:
        """在检索排序顺序中合并来源相邻的候选证据。"""

        candidates: list[ContextCandidate] = []
        for item in evidence:
            if candidates and _is_adjacent(candidates[-1].primary_chunk, item.primary_chunk):
                previous = candidates[-1]
                candidates[-1] = ContextCandidate(
                    text=f"{previous.text}\n{item.text}",
                    evidence=(*previous.evidence, item),
                )
            else:
                candidates.append(ContextCandidate(text=item.text, evidence=(item,)))
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
            for item in candidate.evidence:
                _append_dropped_evidence(item, dropped_chunks, reason=reason, detail=detail)

    @staticmethod
    def _append_used_chunks(
        candidate: ContextCandidate,
        used_chunks: list[RetrievedChunk],
        used_chunk_identities: set[tuple[str, str]],
    ) -> None:
        """将候选段的来源 chunk 去重后写入最终结果。"""

        for chunk in candidate.source_chunks:
            identity = (chunk.chunk_id, chunk.version_id)
            if identity not in used_chunk_identities:
                used_chunk_identities.add(identity)
                used_chunks.append(chunk)

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

        source_chunks = candidate.source_chunks
        page_starts = [
            chunk.page_start for chunk in source_chunks if chunk.page_start is not None
        ]
        page_ends = [
            chunk.page_end for chunk in source_chunks if chunk.page_end is not None
        ]
        sections = tuple(
            dict.fromkeys(chunk.section for chunk in source_chunks if chunk.section is not None)
        )
        return ContextSegment(
            citation_id=citation_id,
            text=text,
            source_chunk_ids=tuple(chunk.chunk_id for chunk in source_chunks),
            source_doc_id=candidate.primary_chunk.doc_id,
            page_start=min(page_starts) if page_starts else None,
            page_end=max(page_ends) if page_ends else None,
            sections=sections,
            token_count=token_count,
            is_truncated=is_truncated,
            source_ranges=_build_source_ranges(
                candidate,
                displayed_text=text,
                is_truncated=is_truncated,
            ),
        )

    @staticmethod
    def _build_citation(segment: ContextSegment, candidate: ContextCandidate) -> Citation:
        """构建与段级 provenance 对应的兼容 citation。"""

        first_chunk = candidate.primary_chunk
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


def _append_dropped_evidence(
    evidence: EvidenceCandidate,
    dropped_chunks: list[DroppedChunk],
    *,
    reason: str,
    detail: str,
) -> None:
    """将一个候选证据关联的所有来源记录为未采用。"""

    for chunk in evidence.source_chunks:
        dropped_chunks.append(
            DroppedChunk(chunk_id=chunk.chunk_id, reason=reason, detail=detail)
        )


def _is_adjacent(left: RetrievedChunk, right: RetrievedChunk) -> bool:
    """判断两个 chunk 是否来自同一文档版本的相邻位置。"""

    return (
        left.doc_id == right.doc_id
        and left.version_id == right.version_id
        and left.chunk_index + 1 == right.chunk_index
    )


def _build_source_ranges(
    candidate: ContextCandidate,
    *,
    displayed_text: str,
    is_truncated: bool,
) -> tuple[ContextSourceRange, ...]:
    """构建最终展示文本对应的精确来源范围。"""

    if is_truncated:
        return _build_truncated_source_ranges(candidate, displayed_text)

    return _build_full_source_ranges(candidate)


def _build_full_source_ranges(
    candidate: ContextCandidate,
) -> tuple[ContextSourceRange, ...]:
    """记录未截断候选的全部原始来源范围。"""

    ranges: list[ContextSourceRange] = []
    seen: set[tuple[str, str, int, int]] = set()
    for evidence in candidate.evidence:
        for source in evidence.sources:
            identity = (
                source.chunk.chunk_id,
                source.chunk.version_id,
                source.char_start,
                source.char_end,
            )
            if identity in seen:
                continue
            seen.add(identity)
            ranges.append(
                ContextSourceRange(
                    chunk_id=source.chunk.chunk_id,
                    version_id=source.chunk.version_id,
                    char_start=source.char_start,
                    char_end=source.char_end,
                )
            )
    return tuple(ranges)


def _build_truncated_source_ranges(
    candidate: ContextCandidate,
    displayed_text: str,
) -> tuple[ContextSourceRange, ...]:
    """仅记录实际展示前缀可以精确映射到原文的位置。"""

    visible_length = len(displayed_text.removesuffix("..."))
    remaining_length = visible_length
    ranges: list[ContextSourceRange] = []

    for evidence_index, evidence in enumerate(candidate.evidence):
        if remaining_length <= 0:
            break

        consumed_length = min(len(evidence.text), remaining_length)
        ranges.extend(
            _build_evidence_prefix_ranges(
                evidence,
                consumed_length=consumed_length,
            )
        )
        remaining_length -= consumed_length
        if consumed_length < len(evidence.text):
            break

        if evidence_index < len(candidate.evidence) - 1:
            remaining_length = max(0, remaining_length - 1)

    return tuple(ranges)


def _build_evidence_prefix_ranges(
    evidence: EvidenceCandidate,
    *,
    consumed_length: int,
) -> tuple[ContextSourceRange, ...]:
    """为可直接映射的单来源证据前缀创建精确范围。"""

    if consumed_length <= 0 or len(evidence.sources) != 1:
        return ()

    source = evidence.sources[0]
    source_text = source.chunk.text[source.char_start : source.char_end]
    if evidence.text != source_text:
        return ()

    return (
        ContextSourceRange(
            chunk_id=source.chunk.chunk_id,
            version_id=source.chunk.version_id,
            char_start=source.char_start,
            char_end=source.char_start + consumed_length,
        ),
    )
