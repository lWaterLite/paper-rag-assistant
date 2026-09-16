"""本地 JSONL 黄金评测集 Repository。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from app.core.errors import AppError, ErrorCode
from app.evaluation.models import EvaluationCase


class EvaluationDatasetRepository(Protocol):
    """评测数据集的持久化读取协议。"""

    def load(self, path: Path) -> tuple[EvaluationCase, ...]:
        """读取并校验完整评测集。"""


class JsonlEvaluationDatasetRepository:
    """读取一行一条评测样例的 JSONL 文件。"""

    def load(self, path: Path) -> tuple[EvaluationCase, ...]:
        """读取数据集，拒绝重复 id、无效 JSON 和不合法结构。"""

        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise AppError(
                ErrorCode.EVALUATION_FAILED,
                f"评测数据集读取失败：{path}",
            ) from exc

        cases: list[EvaluationCase] = []
        seen_case_ids: set[str] = set()
        for line_number, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"评测数据集 JSON 无法解析：{path}:{line_number}"
                ) from exc
            if not isinstance(payload, dict):
                raise TypeError(f"评测数据集记录必须是 JSON 对象：{path}:{line_number}")
            try:
                case = EvaluationCase.from_dict(payload)
            except TypeError as exc:
                raise TypeError(
                    f"评测数据集记录类型无效：{path}:{line_number}；{exc}"
                ) from exc
            except ValueError as exc:
                raise ValueError(
                    f"评测数据集记录取值无效：{path}:{line_number}；{exc}"
                ) from exc
            if case.case_id in seen_case_ids:
                raise ValueError(f"评测数据集存在重复 id：{case.case_id}")
            seen_case_ids.add(case.case_id)
            cases.append(case)
        if not cases:
            raise ValueError(f"评测数据集为空：{path}")
        return tuple(cases)
