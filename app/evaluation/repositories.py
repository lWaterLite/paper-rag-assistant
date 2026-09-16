"""评测运行结果的本地持久化 Repository。"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from app.evaluation.models import (
    EvaluationCaseResult,
    EvaluationRun,
    EvaluationRunArtifacts,
)


class EvaluationResultRepository(Protocol):
    """持久化单次评测运行及其样例级事实。"""

    def write(self, run: EvaluationRun, run_dir: Path) -> EvaluationRunArtifacts:
        """写入结构化运行产物；调用方必须提前准备运行目录。"""


class LocalJsonEvaluationResultRepository:
    """将运行摘要、样例结果和失败结果分别写入稳定 JSON 产物。"""

    def write(self, run: EvaluationRun, run_dir: Path) -> EvaluationRunArtifacts:
        """写入 JSON 与 JSONL 产物，不承担目录准备职责。"""

        run_path = run_dir / "run.json"
        case_results_path = run_dir / "case_results.jsonl"
        summary_path = run_dir / "summary.json"
        failures_path = run_dir / "failures.jsonl"

        run_path.write_text(
            json.dumps(_serialize_run_metadata(run), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _write_jsonl(
            case_results_path,
            (_serialize_case_result(result) for result in run.case_results),
        )
        summary_path.write_text(
            json.dumps(asdict(run.summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _write_jsonl(
            failures_path,
            (
                _serialize_case_result(result)
                for result in run.case_results
                if result.failure is not None
            ),
        )
        return EvaluationRunArtifacts(
            run_dir=run_dir.as_posix(),
            run_path=run_path.as_posix(),
            case_results_path=case_results_path.as_posix(),
            summary_path=summary_path.as_posix(),
            failures_path=failures_path.as_posix(),
            report_path=(run_dir / "EVALUATION.md").as_posix(),
        )


def _serialize_run_metadata(run: EvaluationRun) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": run.run_id,
        "dataset": {"id": run.dataset_id, "version": run.dataset_version},
        "experiment": {
            "name": run.experiment_name,
            "snapshot": run.experiment_snapshot,
        },
        "index": asdict(run.index_snapshot),
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "summary_path": "summary.json",
        "case_results_path": "case_results.jsonl",
        "failures_path": "failures.jsonl",
    }


def _serialize_case_result(result: EvaluationCaseResult) -> dict[str, object]:
    return asdict(result)


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True)
        for record in records
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
