"""OpenAI-compatible Chat Completions HTTP 适配器。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.llm.config import LlmClientConfig
from app.llm.models import LlmRequest, LlmResponse, LlmUsage


class LlmClientRequestError(RuntimeError):
    """LLM 供应商请求失败，供上层转换为领域错误。"""

    def __init__(self, message: str, *, retriable: bool = False) -> None:
        super().__init__(message)
        self.retriable = retriable


@dataclass(frozen=True, slots=True)
class OpenAiCompatibleLlmClient:
    """通过标准库调用兼容 Chat Completions 协议的服务。

    `base_url` 必须指向完整的 chat completions endpoint。该实现不依赖供应商 SDK，
    从而保持业务层和第三方客户端对象解耦。
    """

    config: LlmClientConfig
    api_key: str

    def __post_init__(self) -> None:
        if not self.config.base_url:
            raise ValueError("openai_compatible provider 需要配置 base_url")
        if not self.api_key.strip():
            raise ValueError("openai_compatible provider 需要 RAG_LLM_API_KEY")

    @property
    def provider_name(self) -> str:
        """返回注册表使用的稳定名称。"""

        return "openai_compatible"

    def complete(self, request: LlmRequest) -> LlmResponse:
        """发送一次 Chat Completions 请求并归一化响应。"""

        endpoint = self.config.base_url
        if endpoint is None:
            raise LlmClientRequestError("LLM endpoint 未配置")
        payload = {
            "model": request.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }
        http_request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(http_request, timeout=request.timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
                response_id = response.headers.get("x-request-id")
        except HTTPError as exc:
            raise LlmClientRequestError(
                f"LLM 服务返回 HTTP {exc.code}",
                retriable=exc.code == 429 or 500 <= exc.code < 600,
            ) from exc
        except URLError as exc:
            raise LlmClientRequestError(
                f"LLM 服务连接失败：{exc.reason}",
                retriable=True,
            ) from exc
        except OSError as exc:
            raise LlmClientRequestError(f"LLM 请求失败：{exc}", retriable=True) from exc

        try:
            data = json.loads(raw_body)
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise LlmClientRequestError("LLM 响应不符合 Chat Completions 协议") from exc

        if not isinstance(content, str) or not content.strip():
            raise LlmClientRequestError("LLM 返回了空内容")

        usage = _parse_usage(data.get("usage"))
        return LlmResponse(
            content=content,
            model=str(data.get("model") or request.model),
            provider_request_id=str(data.get("id") or response_id) if (data.get("id") or response_id) else None,
            finish_reason=(
                str(choice["finish_reason"])
                if choice.get("finish_reason") is not None
                else None
            ),
            usage=usage,
        )


def _parse_usage(value: Any) -> LlmUsage:
    """从兼容响应中安全读取 token 用量。"""

    if not isinstance(value, dict):
        return LlmUsage()
    input_tokens = value.get("prompt_tokens")
    output_tokens = value.get("completion_tokens")
    return LlmUsage(
        input_tokens=input_tokens if isinstance(input_tokens, int) else None,
        output_tokens=output_tokens if isinstance(output_tokens, int) else None,
    )
