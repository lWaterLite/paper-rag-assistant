"""评测子系统运行时 Config。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """一次实验可变部分的不可变快照。"""

    name: str
    retriever: str
    top_k: int
    reranking_enabled: bool

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("experiment 名称不能为空")
        if not self.retriever.strip():
            raise ValueError("experiment retriever 不能为空")
        if self.top_k <= 0:
            raise ValueError("experiment top_k 必须大于 0")


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """评测运行器使用的不可变配置。"""

    dataset_path: Path
    dataset_id: str
    dataset_version: str
    output_dir: Path
    include_answer_text: bool
    max_cases: int | None
    experiments: tuple[ExperimentConfig, ...]

    def __post_init__(self) -> None:
        if not self.dataset_id.strip() or not self.dataset_version.strip():
            raise ValueError("评测数据集标识与版本不能为空")
        if self.max_cases is not None and self.max_cases <= 0:
            raise ValueError("max_cases 必须大于 0")
        if not self.experiments:
            raise ValueError("至少需要一个评测 experiment")
