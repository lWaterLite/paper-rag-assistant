"""检索后处理配置的跨字段校验。"""

from __future__ import annotations

from app.retrieval.postprocessing.config import PostProcessingConfig


class PostProcessingConfigValidator:
    """验证多个运行时 Config 组合后的领域约束。"""

    @staticmethod
    def validate(config: PostProcessingConfig) -> None:
        """校验组合是否能够产生有意义的后处理流程。

        每个局部 Config 已在自身构造时验证字段类型与单字段范围；这里仅处理
        需要同时观察 retrieval、reranking 和 context packing 的规则。
        """

        errors: list[str] = []
        reranking = config.reranking
        retrieval = config.retrieval
        context_packing = config.context_packing

        if reranking.enabled and reranking.candidate_limit < retrieval.top_k:
            errors.append(
                "retrieval.reranking.candidate_limit 必须大于等于 "
                "retrieval.top_k：启用 rerank 时，候选宽度不能小于最终返回数量"
            )

        static_available_context_tokens = (
            context_packing.model_context_window
            - context_packing.reserved_prompt_tokens
            - context_packing.reserved_output_tokens
            - context_packing.safety_margin_tokens
        )
        if static_available_context_tokens <= 0:
            errors.append(
                "retrieval.context_packing.model_context_window 必须大于 "
                "reserved_prompt_tokens、reserved_output_tokens 与 "
                "safety_margin_tokens 之和：当前不存在可用于资料上下文的 token 预算"
            )
        elif context_packing.max_context_tokens > static_available_context_tokens:
            errors.append(
                "retrieval.context_packing.max_context_tokens 不能大于在空问题下的 "
                "静态可用资料预算：请减少 max_context_tokens 或调整模型窗口与预留 token"
            )

        if errors:
            raise ValueError("后处理流程配置无效：" + "；".join(errors))
