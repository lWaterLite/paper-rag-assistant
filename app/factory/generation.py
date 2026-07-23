"""回答生成与引用校验对象组装。"""

from __future__ import annotations

from dataclasses import dataclass

from app.factory.configs import ConfigFactory
from app.factory.llm import LlmFactory
from app.factory.retrieval import RetrievalFactory
from app.generation.answering import AnswerGenerator, GroundedAnswerGenerator
from app.generation.answering.budget import PromptBudgetValidator
from app.generation.citations import CitationValidator
from app.generation.prompts import RagAnswerPromptBuilder


@dataclass(slots=True)
class GenerationFactory:
    """组合回答生成器需要的 prompt、token 与 citation 依赖。"""

    configs: ConfigFactory
    retrieval: RetrievalFactory
    llm: LlmFactory

    def build_answer_generator(self) -> AnswerGenerator:
        """创建受当前证据、预算和引用规则约束的生成器。"""

        generation_config = self.configs.generation.answering
        return GroundedAnswerGenerator(
            config=generation_config,
            llm_client=self.llm.client,
            prompt_builder=RagAnswerPromptBuilder(generation_config),
            citation_validator=CitationValidator(
                self.configs.generation.citation_validation
            ),
            budget_validator=PromptBudgetValidator(
                context_config=self.configs.retrieval.context_packer,
                token_estimator=self.retrieval.build_token_estimator(),
            ),
        )
