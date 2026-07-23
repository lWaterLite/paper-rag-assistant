"""项目配置聚合与 TOML 加载入口。"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from app.core.errors import AppError, ErrorCode
from app.core.settings.indexing import IndexingSettings
from app.core.settings.ingestion import IngestionSettings
from app.core.settings.retrieval import RetrievalSettings
from app.core.settings.generation import GenerationSettings


class ProjectSettings(BaseModel):
    """从 settings.toml 读取的结构化工程配置。"""

    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    indexing: IndexingSettings = Field(default_factory=IndexingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)

    @classmethod
    def from_toml(cls, path: Path | str = Path("settings.toml")) -> "ProjectSettings":
        """从 TOML 文件读取结构化配置。"""

        config_path = Path(path)
        if not config_path.exists():
            return cls()

        try:
            data = tomllib.loads(config_path.read_text(encoding="utf-8"))
            return cls.model_validate(data)
        except (OSError, tomllib.TOMLDecodeError, ValidationError) as exc:
            if isinstance(exc, ValidationError):
                message = format_validation_error(exc)
            else:
                message = f"配置文件读取失败：{exc}"
            raise AppError(ErrorCode.INVALID_CONFIG, message) from exc


def format_validation_error(error: ValidationError) -> str:
    """把 Pydantic 校验错误整理成适合 CLI/API 展示的中文消息。"""

    messages: list[str] = []
    for item in error.errors():
        location = ".".join(str(part) for part in item.get("loc", ())) or "settings"
        messages.append(f"{location}: {item.get('msg')}")
    return "配置校验失败：" + "；".join(messages)
