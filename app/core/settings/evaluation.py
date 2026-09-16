"""RAG 离线评测的外部 Settings。"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class EvaluationExperimentSettings(BaseModel):
    """一个可比较评测 profile 的外部配置。"""

    retriever: str = Field(min_length=1, description="本次实验使用的 retriever 策略")
    top_k: int = Field(gt=0, description="本次实验的最终检索结果数量")
    reranking_enabled: bool = Field(description="是否启用 rerank 后处理")

    @model_validator(mode="after")
    def validate_retriever(self) -> EvaluationExperimentSettings:
        """清理策略名；实际合法性仍由 RetrieverRegistry 负责。"""

        self.retriever = self.retriever.strip()
        if not self.retriever:
            raise ValueError("retriever 不能为空")
        return self


class EvaluationSettings(BaseModel):
    """离线评测数据集、运行产物与实验 profiles 的外部配置。"""

    dataset_path: Path = Field(
        default=Path("evaluation/datasets/papers_eval_v1.jsonl"),
        description="人工维护的黄金评测集 JSONL 路径",
    )
    dataset_id: str = Field(default="papers_eval", min_length=1)
    dataset_version: str = Field(default="v1", min_length=1)
    output_dir: Path = Field(default=Path("evaluation/runs"))
    include_answer_text: bool = Field(
        default=True,
        description="是否在离线评测产物中保存回答正文；公开导出前应按数据策略审查",
    )
    max_cases: int | None = Field(
        default=None,
        gt=0,
        description="开发调试时限制运行样例数；null 表示运行完整数据集",
    )
    experiments: dict[str, EvaluationExperimentSettings] = Field(
        default_factory=lambda: {
            "vector_baseline": EvaluationExperimentSettings(
                retriever="vector", top_k=5, reranking_enabled=False
            ),
            "vector_rerank": EvaluationExperimentSettings(
                retriever="vector", top_k=5, reranking_enabled=True
            ),
            "hybrid_rerank": EvaluationExperimentSettings(
                retriever="hybrid", top_k=5, reranking_enabled=True
            ),
        }
    )

    @model_validator(mode="after")
    def validate_identifiers(self) -> EvaluationSettings:
        """校验数据集与实验 profile 标识，避免不可追溯的运行产物。"""

        self.dataset_id = self.dataset_id.strip()
        self.dataset_version = self.dataset_version.strip()
        if not self.dataset_id or not self.dataset_version:
            raise ValueError("dataset_id 与 dataset_version 不能为空")
        normalized_experiments: dict[str, EvaluationExperimentSettings] = {}
        for name, experiment in self.experiments.items():
            normalized_name = name.strip()
            if not normalized_name:
                raise ValueError("evaluation experiments 中不能包含空名称")
            if normalized_name in normalized_experiments:
                raise ValueError(f"evaluation experiment 名称重复：{normalized_name}")
            normalized_experiments[normalized_name] = experiment
        if not normalized_experiments:
            raise ValueError("至少需要配置一个 evaluation experiment")
        self.experiments = normalized_experiments
        return self
