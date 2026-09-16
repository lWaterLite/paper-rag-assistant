"""Evaluation Settings 到运行时 Config 的适配器。"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.settings.evaluation import EvaluationSettings
from app.evaluation.configuration import EvaluationConfig, ExperimentConfig


@dataclass(frozen=True, slots=True)
class EvaluationConfigAdapter:
    """将外部评测配置转换为评测领域需要的不可变快照。"""

    settings: EvaluationSettings
    evaluation: EvaluationConfig = field(init=False)

    def __post_init__(self) -> None:
        experiments = tuple(
            ExperimentConfig(
                name=name,
                retriever=experiment.retriever,
                top_k=experiment.top_k,
                reranking_enabled=experiment.reranking_enabled,
            )
            for name, experiment in self.settings.experiments.items()
        )
        object.__setattr__(
            self,
            "evaluation",
            EvaluationConfig(
                dataset_path=self.settings.dataset_path,
                dataset_id=self.settings.dataset_id,
                dataset_version=self.settings.dataset_version,
                output_dir=self.settings.output_dir,
                include_answer_text=self.settings.include_answer_text,
                max_cases=self.settings.max_cases,
                experiments=experiments,
            ),
        )
