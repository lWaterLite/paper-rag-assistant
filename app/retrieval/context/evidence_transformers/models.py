"""候选证据变换的输入、输出与来源追溯模型。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from app.core.models import RetrievedChunk


@dataclass(frozen=True, slots=True)
class EvidenceSource:
    """候选证据在一个原始检索 chunk 中的可追溯来源范围。"""

    chunk: RetrievedChunk
    char_start: int
    char_end: int

    def __post_init__(self) -> None:
        if self.char_start < 0:
            raise ValueError("evidence source 的 char_start 不能小于 0")
        if self.char_end <= self.char_start:
            raise ValueError("evidence source 的 char_end 必须大于 char_start")
        if self.char_end > len(self.chunk.text):
            raise ValueError("evidence source 的字符范围超出原始 chunk 文本长度")

    @classmethod
    def full_chunk(cls, chunk: RetrievedChunk) -> "EvidenceSource":
        """创建覆盖整个原始 chunk 的来源范围。"""

        if not chunk.text:
            raise ValueError("不能从空文本 chunk 创建 evidence source")
        return cls(chunk=chunk, char_start=0, char_end=len(chunk.text))


@dataclass(frozen=True, slots=True)
class EvidenceCandidate:
    """进入 ContextPacker 前的可追溯候选证据。

    一个候选可以来自单个 chunk 的局部范围，也可以在未来来自多个 chunk。文本本身
    可以被抽取式变换，但 sources 始终保留对原始检索结果的定位信息。
    """

    evidence_id: str
    text: str
    sources: tuple[EvidenceSource, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_id = self.evidence_id.strip()
        if not normalized_id:
            raise ValueError("evidence_id 不能为空")
        if not self.text.strip():
            raise ValueError("evidence candidate 的文本不能为空")
        if not self.sources:
            raise ValueError("evidence candidate 必须包含至少一个来源")
        object.__setattr__(self, "evidence_id", normalized_id)

    @property
    def primary_chunk(self) -> RetrievedChunk:
        """返回用于排序、文档配额和 Citation 兼容字段的首个来源。"""

        return self.sources[0].chunk

    @property
    def source_chunks(self) -> tuple[RetrievedChunk, ...]:
        """按首次出现顺序返回去重后的原始来源 chunk。"""

        chunks_by_identity: dict[tuple[str, str], RetrievedChunk] = {}
        for source in self.sources:
            key = (source.chunk.chunk_id, source.chunk.version_id)
            chunks_by_identity.setdefault(key, source.chunk)
        return tuple(chunks_by_identity.values())

    @classmethod
    def from_retrieved_chunk(cls, chunk: RetrievedChunk) -> "EvidenceCandidate":
        """把未变换的检索结果包装成完整来源的候选证据。"""

        return cls(
            evidence_id=f"evidence_{chunk.chunk_id}",
            text=chunk.text,
            sources=(EvidenceSource.full_chunk(chunk),),
        )


@dataclass(frozen=True, slots=True)
class EvidenceTransformRequest:
    """EvidenceTransformer 的结构化输入。"""

    query: str
    chunks: Sequence[RetrievedChunk]


@dataclass(frozen=True, slots=True)
class EvidenceTransformResult:
    """具体 transformer 返回的候选证据集合。"""

    candidates: tuple[EvidenceCandidate, ...]


@dataclass(frozen=True, slots=True)
class EvidenceTransformStageResult:
    """EvidenceTransformStage 的结果与可观测性明细。"""

    candidates: tuple[EvidenceCandidate, ...]
    detail: dict[str, Any]


def passthrough_candidates(
    chunks: Sequence[RetrievedChunk],
) -> tuple[EvidenceCandidate, ...]:
    """将原始检索结果转换为完整文本的候选证据。"""

    return tuple(EvidenceCandidate.from_retrieved_chunk(chunk) for chunk in chunks)
