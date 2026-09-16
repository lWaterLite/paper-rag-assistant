"""评测数据集、单样例结果与运行汇总的领域模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, cast

QuestionType = Literal[
    "fact",
    "compare",
    "summary",
    "citation_lookup",
    "abstention",
    "ambiguous",
]
EvaluationResultStatus = Literal["success", "error"]

_VALID_QUESTION_TYPES: frozenset[str] = frozenset(
    {"fact", "compare", "summary", "citation_lookup", "abstention", "ambiguous"}
)


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """一条黄金评测样例及其人工维护的期望证据。"""

    case_id: str
    question: str
    question_type: QuestionType
    expected_doc_ids: tuple[str, ...] = ()
    expected_chunk_ids: tuple[str, ...] = ()
    expected_source_paths: tuple[str, ...] = ()
    expected_keywords: tuple[str, ...] = ()
    answer_notes: str = ""
    expected_abstention: bool = False
    tags: tuple[str, ...] = ()
    split: str = "development"

    def __post_init__(self) -> None:
        case_id = self.case_id.strip()
        question = self.question.strip()
        if not case_id:
            raise ValueError("评测样例 id 不能为空")
        if not question:
            raise ValueError(f"评测样例问题不能为空：{case_id}")
        if self.question_type not in _VALID_QUESTION_TYPES:
            allowed = ", ".join(sorted(_VALID_QUESTION_TYPES))
            raise ValueError(f"评测样例类型非法：{self.question_type}；可选值：{allowed}")
        if not self.expected_abstention and not (
            self.expected_doc_ids
            or self.expected_chunk_ids
            or self.expected_source_paths
        ):
            raise ValueError(
                f"非拒答评测样例必须提供期望文档或 chunk：{case_id}"
            )
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "question", question)
        object.__setattr__(self, "expected_doc_ids", _normalize_values(self.expected_doc_ids))
        object.__setattr__(
            self,
            "expected_chunk_ids",
            _normalize_values(self.expected_chunk_ids),
        )
        object.__setattr__(
            self,
            "expected_source_paths",
            _normalize_values(self.expected_source_paths),
        )
        object.__setattr__(
            self,
            "expected_keywords",
            _normalize_values(self.expected_keywords),
        )
        object.__setattr__(self, "tags", _normalize_values(self.tags))
        object.__setattr__(self, "answer_notes", self.answer_notes.strip())
        object.__setattr__(self, "split", self.split.strip() or "development")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationCase:
        """从 JSONL 记录恢复并校验领域样例。"""

        return cls(
            case_id=_required_string(data, "id"),
            question=_required_string(data, "question"),
            question_type=cast(QuestionType, _required_string(data, "question_type")),
            expected_doc_ids=_optional_string_sequence(data, "expected_doc_ids"),
            expected_chunk_ids=_optional_string_sequence(data, "expected_chunk_ids"),
            expected_source_paths=_optional_string_sequence(
                data,
                "expected_source_paths",
            ),
            expected_keywords=_optional_string_sequence(data, "expected_keywords"),
            answer_notes=_optional_string(data, "answer_notes"),
            expected_abstention=_optional_bool(data, "expected_abstention", False),
            tags=_optional_string_sequence(data, "tags"),
            split=_optional_string(data, "split") or "development",
        )


@dataclass(frozen=True, slots=True)
class EvaluationChunkObservation:
    """评测运行中保留的检索证据身份与排序摘要。"""

    chunk_id: str
    doc_id: str
    version_id: str
    rank: int
    score: float
    retriever: str


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """单样例可自动计算的检索、上下文、引用与拒答指标。"""

    retrieval_hit_rate: float | None = None
    retrieval_recall: float | None = None
    retrieval_reciprocal_rank: float | None = None
    retrieval_precision: float | None = None
    context_recall: float | None = None
    context_precision: float | None = None
    citation_coverage: float | None = None
    answer_has_citation: float | None = None
    abstention_correct: float | None = None

    def as_mapping(self) -> dict[str, float | None]:
        """返回便于聚合的稳定指标映射。"""

        return {
            "retrieval_hit_rate": self.retrieval_hit_rate,
            "retrieval_recall": self.retrieval_recall,
            "retrieval_mrr": self.retrieval_reciprocal_rank,
            "retrieval_precision": self.retrieval_precision,
            "context_recall": self.context_recall,
            "context_precision": self.context_precision,
            "citation_coverage": self.citation_coverage,
            "answer_has_citation_ratio": self.answer_has_citation,
            "abstention_accuracy": self.abstention_correct,
        }


@dataclass(frozen=True, slots=True)
class ManualReview:
    """人工或未来 judge 写入的语义质量标注，保持与自动指标分离。"""

    answer_relevance: float | None = None
    faithfulness: float | None = None
    groundedness: float | None = None
    citation_correctness: float | None = None
    failure_type: str | None = None
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationFailure:
    """单样例运行失败的可审计摘要。"""

    failure_type: str
    error_code: str | None
    message: str


@dataclass(frozen=True, slots=True)
class EvaluationCaseResult:
    """一个评测样例在某个实验配置下的完整事实记录。"""

    case: EvaluationCase
    status: EvaluationResultStatus
    retrieved_chunks: tuple[EvaluationChunkObservation, ...] = ()
    context_chunk_ids: tuple[str, ...] = ()
    context_doc_ids: tuple[str, ...] = ()
    context_source_paths: tuple[str, ...] = ()
    dropped_chunk_ids: tuple[str, ...] = ()
    answer_status: str | None = None
    answer_text: str | None = None
    answer_citation_ids: tuple[str, ...] = ()
    answer_citation_chunk_ids: tuple[str, ...] = ()
    answer_citation_doc_ids: tuple[str, ...] = ()
    answer_citation_source_paths: tuple[str, ...] = ()
    abstention_reason: str | None = None
    trace_id: str | None = None
    latency_ms: float | None = None
    metrics: EvaluationMetrics = field(default_factory=EvaluationMetrics)
    review: ManualReview | None = None
    failure: EvaluationFailure | None = None

    def __post_init__(self) -> None:
        if self.status == "success" and self.failure is not None:
            raise ValueError("成功的评测结果不能携带 failure")
        if self.status == "error" and self.failure is None:
            raise ValueError("失败的评测结果必须携带 failure")


@dataclass(frozen=True, slots=True)
class EvaluationIndexSnapshot:
    """本次实验绑定的索引身份，避免跨版本误比较。"""

    index_id: str
    artifact_definition_hash: str
    document_set_hash: str
    document_count: int
    chunk_count: int
    vector_count: int


@dataclass(frozen=True, slots=True)
class MetricAggregate:
    """一个指标的宏平均值及有效样例数量。"""

    value: float | None
    evaluated_case_count: int


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """一次实验的汇总指标、问题类型分组与失败计数。"""

    total_case_count: int
    success_case_count: int
    error_case_count: int
    metrics: dict[str, MetricAggregate]
    metrics_by_question_type: dict[str, dict[str, MetricAggregate]]
    failure_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    """一个数据集、索引和实验配置组成的可复现实验运行。"""

    run_id: str
    dataset_id: str
    dataset_version: str
    experiment_name: str
    experiment_snapshot: dict[str, object]
    index_snapshot: EvaluationIndexSnapshot
    case_results: tuple[EvaluationCaseResult, ...]
    summary: EvaluationSummary
    started_at: str
    completed_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass(frozen=True, slots=True)
class EvaluationRunArtifacts:
    """一次运行写入的持久化产物路径。"""

    run_dir: str
    run_path: str
    case_results_path: str
    summary_path: str
    failures_path: str
    report_path: str


@dataclass(frozen=True, slots=True)
class EvaluationRunResult:
    """评测运行的领域结果与产物位置。"""

    run: EvaluationRun
    artifacts: EvaluationRunArtifacts


def _normalize_values(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError("评测样例中的标识和标签必须是字符串")
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            normalized.append(cleaned)
    return tuple(normalized)


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise TypeError(f"评测样例字段必须是字符串：{key}")
    if not value.strip():
        raise ValueError(f"评测样例字段不能为空：{key}")
    return value


def _optional_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key, "")
    if not isinstance(value, str):
        raise TypeError(f"评测样例字段必须是字符串：{key}")
    return value


def _optional_string_sequence(data: dict[str, Any], key: str) -> tuple[str, ...]:
    value = data.get(key, [])
    if not isinstance(value, list):
        raise TypeError(f"评测样例字段必须是字符串列表：{key}")
    if not all(isinstance(item, str) for item in value):
        raise TypeError(f"评测样例字段必须是字符串列表：{key}")
    return tuple(value)


def _optional_bool(data: dict[str, Any], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"评测样例字段必须是布尔值：{key}")
    return value
