"""顶层 RAG Pipeline Settings 到运行时 Config 的适配。"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.settings import RetrievalSettings
from app.pipeline import RagPipelineConfig


@dataclass(frozen=True, slots=True)
class PipelineConfigAdapter:
    """从顶层相关 Settings 生成在线 RAG Pipeline Config。"""

    retrieval_settings: RetrievalSettings
    rag_pipeline: RagPipelineConfig = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rag_pipeline",
            RagPipelineConfig(top_k=self.retrieval_settings.top_k),
        )
