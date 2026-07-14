"""跨模块共享的配置、错误、元数据与追踪能力。"""

from app.core.tracing import PipelineStageRun, RagTrace

__all__ = ["PipelineStageRun", "RagTrace"]
