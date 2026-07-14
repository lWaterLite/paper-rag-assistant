"""检索后处理流程的运行时配置聚合。"""

from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.configuration.retrieval import RetrievalConfig
from app.retrieval.context.packer import ContextPackerConfig
from app.retrieval.context.evidence_transformers.config import (
    EvidenceTransformationConfig,
)
from app.retrieval.rerankers.config import RerankingConfig


@dataclass(frozen=True, slots=True)
class PostProcessingConfig:
    """聚合会共同影响检索后处理行为的运行时配置。

    这个对象不替代各功能模块原有的 Config。它只提供跨模块组合校验所需的
    视图，避免把互相依赖的规则散落到 rerank 或 context packing 的局部配置中。
    """

    retrieval: RetrievalConfig
    reranking: RerankingConfig
    context_packing: ContextPackerConfig
    evidence_transformation: EvidenceTransformationConfig
