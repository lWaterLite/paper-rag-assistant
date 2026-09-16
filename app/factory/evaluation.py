"""离线评测对象组装。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.evaluation.configuration import ExperimentConfig
from app.evaluation.datasets import JsonlEvaluationDatasetRepository
from app.evaluation.reporting import EvaluationReportWriter
from app.evaluation.repositories import LocalJsonEvaluationResultRepository
from app.evaluation.runner import (
    EvaluationRagService,
    EvaluationRunner,
    EvaluationSuiteRunner,
)
from app.factory.configs import ConfigFactory
from app.indexing.pipeline.types import RagIndex


@dataclass(slots=True)
class EvaluationFactory:
    """组装评测 runner 与持久化依赖，不直接构造生产 RAG pipeline。"""

    configs: ConfigFactory

    def build_runner(self) -> EvaluationRunner:
        """创建使用当前评测 Config 的离线运行器。"""

        return EvaluationRunner(
            config=self.configs.evaluation.evaluation,
            dataset_repository=JsonlEvaluationDatasetRepository(),
            result_repository=LocalJsonEvaluationResultRepository(),
            report_writer=EvaluationReportWriter(),
        )

    def build_suite_runner(
        self,
        *,
        index: RagIndex,
        target_builder: Callable[[ExperimentConfig], EvaluationRagService],
    ) -> EvaluationSuiteRunner:
        """构建在同一索引上运行全部 experiment profile 的协调器。"""

        return EvaluationSuiteRunner(
            runner=self.build_runner(),
            index=index,
            target_builder=target_builder,
        )
