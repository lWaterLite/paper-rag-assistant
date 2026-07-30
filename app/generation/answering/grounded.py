"""受证据约束的结构化回答生成器。"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.errors import AppError, ErrorCode
from app.core.tracing import RagTrace
from app.generation.answering.budget import PromptBudgetValidator
from app.generation.citations import CitationValidationError, CitationValidator
from app.generation.configuration import GenerationConfig
from app.generation.generated import GeneratedAnswerPayload
from app.generation.models import Citation, GenerationDiagnostics, RagAnswer
from app.generation.prompts import RagAnswerPromptBuilder
from app.llm import LlmClient, LlmRequest
from app.retrieval.context import PackedContext
from app.retrieval.models import RetrievedChunk


@dataclass(frozen=True, slots=True)
class GroundedAnswerGenerator:
    """将 LLM 输出限制为当前 PackedContext 支持的回答。"""

    config: GenerationConfig
    llm_client: LlmClient
    prompt_builder: RagAnswerPromptBuilder
    citation_validator: CitationValidator
    budget_validator: PromptBudgetValidator

    def generate(
        self,
        question: str,
        packed_context: PackedContext,
        retrieved_chunks: list[RetrievedChunk],
        trace: RagTrace,
    ) -> RagAnswer:
        """生成、解析并校验一次结构化 RAG 回答。"""

        if not packed_context.citations:
            return self._build_abstained_answer(
                retrieved_chunks=retrieved_chunks,
                trace=trace,
                reason="当前知识库中没有检索到足够相关的资料",
                validation_status="not_required_no_evidence",
            )

        prompt = self.prompt_builder.build(question, packed_context)
        try:
            budget_usage = self.budget_validator.validate(
                prompt,
                max_output_tokens=self.config.max_output_tokens,
            )
            response = self.llm_client.complete(
                LlmRequest(
                    messages=prompt.messages,
                    model=self.config.model,
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_output_tokens,
                    timeout_seconds=self.config.timeout_seconds,
                    metadata={
                        "task": "answer_generation",
                        "question": question,
                        "primary_citation_id": packed_context.citations[0].citation_id,
                    },
                )
            )
            payload = GeneratedAnswerPayload.from_json(response.content)
            validation = self.citation_validator.validate(
                payload, packed_context.citations
            )
        except (ValueError, CitationValidationError, TypeError) as exc:
            if self.config.invalid_output_mode == "abstain":
                return self._build_abstained_answer(
                    retrieved_chunks=retrieved_chunks,
                    trace=trace,
                    reason="模型输出未通过回答引用校验",
                    validation_status="invalid_output_abstained",
                )
            raise AppError(
                ErrorCode.CITATION_VALIDATION_FAILED,
                f"回答生成或引用校验失败：{exc}",
            ) from exc
        except Exception as exc:
            raise AppError(ErrorCode.GENERATION_FAILED, f"LLM 调用失败：{exc}") from exc

        citations_by_id = {
            citation.citation_id: Citation.from_context_citation(citation)
            for citation in packed_context.citations
        }
        citations = [
            citations_by_id[citation_id] for citation_id in validation.citation_ids
        ]
        return RagAnswer(
            answer=payload.answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            trace_id=trace.trace_id,
            latency_ms=trace.latency_ms,
            status="abstained" if payload.abstained else "answered",
            abstention_reason=payload.abstention_reason,
            diagnostics=GenerationDiagnostics(
                provider=self.llm_client.provider_name,
                model=response.model,
                prompt_tokens=budget_usage.prompt_tokens,
                output_tokens=response.usage.output_tokens,
                citation_validation="valid",
                provider_request_id=response.provider_request_id,
                finish_reason=response.finish_reason,
            ),
        )

    def _build_abstained_answer(
        self,
        *,
        retrieved_chunks: list[RetrievedChunk],
        trace: RagTrace,
        reason: str,
        validation_status: str,
    ) -> RagAnswer:
        """构造明确的业务性拒答，而不是把资料不足视为技术失败。"""

        return RagAnswer(
            answer=f"根据当前知识库资料无法确定：{reason}。",
            citations=[],
            retrieved_chunks=retrieved_chunks,
            trace_id=trace.trace_id,
            latency_ms=trace.latency_ms,
            status="abstained",
            abstention_reason=reason,
            diagnostics=GenerationDiagnostics(
                provider=self.llm_client.provider_name,
                model=self.config.model,
                prompt_tokens=0,
                output_tokens=None,
                citation_validation=validation_status,
            ),
        )
