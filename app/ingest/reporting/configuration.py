"""文档摄取报告的运行时配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IngestionReportConfig:
    """摄取报告写入位置。"""

    output_dir: Path = Path("logs")

    @property
    def output_path(self) -> Path:
        """返回默认摄取报告路径。"""

        return self.output_dir / "ingestion_report.json"


@dataclass(frozen=True)
class ChunkingReportConfig:
    """切分质量报告写入位置。"""

    output_dir: Path = Path("logs")

    @property
    def output_path(self) -> Path:
        """返回默认切分质量报告路径。"""

        return self.output_dir / "chunking_report.json"
