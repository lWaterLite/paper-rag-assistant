"""评测运行的 Markdown 报告写入器。"""

from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.models import EvaluationRun


class EvaluationReportWriter:
    """将一次运行的汇总和失败案例写为可审阅 Markdown。"""

    def write(self, run: EvaluationRun, output_path: Path) -> Path:
        """写入当前运行的 EVALUATION.md；调用方必须提前准备目录。"""

        output_path.write_text(self.build_markdown(run), encoding="utf-8")
        return output_path

    @staticmethod
    def build_markdown(run: EvaluationRun) -> str:
        """构建只包含运行事实的 Markdown 报告。"""

        lines = [
            f"# Evaluation Run: {run.run_id}",
            "",
            "## 运行身份",
            "",
            f"- 数据集：`{run.dataset_id}` / `{run.dataset_version}`",
            f"- 实验：`{run.experiment_name}`",
            f"- 索引：`{run.index_snapshot.index_id}`",
            f"- 开始时间：`{run.started_at}`",
            f"- 完成时间：`{run.completed_at}`",
            "",
            "## 实验配置",
            "",
            "```json",
            _render_json(run.experiment_snapshot),
            "```",
            "",
            "## 汇总",
            "",
            f"- 总样例：{run.summary.total_case_count}",
            f"- 成功：{run.summary.success_case_count}",
            f"- 失败：{run.summary.error_case_count}",
            "",
            "## 自动指标",
            "",
            "| 指标 | 数值 | 有效样例数 |",
            "| --- | ---: | ---: |",
        ]
        for name, aggregate in sorted(run.summary.metrics.items()):
            value = "N/A" if aggregate.value is None else f"{aggregate.value:.4f}"
            lines.append(f"| `{name}` | {value} | {aggregate.evaluated_case_count} |")

        lines.extend(["", "## 按问题类型", ""])
        for question_type, metrics in run.summary.metrics_by_question_type.items():
            lines.append(f"### `{question_type}`")
            lines.append("")
            lines.append("| 指标 | 数值 | 有效样例数 |")
            lines.append("| --- | ---: | ---: |")
            for name, aggregate in sorted(metrics.items()):
                value = "N/A" if aggregate.value is None else f"{aggregate.value:.4f}"
                lines.append(
                    f"| `{name}` | {value} | {aggregate.evaluated_case_count} |"
                )
            lines.append("")

        lines.extend(["## 失败案例", ""])
        failures = [result for result in run.case_results if result.failure is not None]
        if not failures:
            lines.append("本次运行没有技术失败样例。")
        else:
            lines.extend(["| 样例 | 类型 | 失败阶段 | 说明 |", "| --- | --- | --- | --- |"])
            for result in failures:
                assert result.failure is not None
                message = result.failure.message.replace("|", "\\|").replace("\n", " ")
                lines.append(
                    f"| `{result.case.case_id}` | `{result.case.question_type}` | "
                    f"`{result.failure.failure_type}` | {message} |"
                )
        lines.extend(
            [
                "",
                "## 解释边界",
                "",
                (
                    "本报告中的检索、上下文、citation 与拒答指标均为确定性运行事实。"
                    "回答相关性、faithfulness、groundedness 与 citation correctness 需要通过"
                    "人工评审或独立 judge rubric 追加到样例级结果，不能仅由本报告自动推断。"
                ),
                "",
            ]
        )
        return "\n".join(lines)


def _render_json(value: dict[str, object]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
