"""DeepSeek · Chat Completions 通路（本仓默认）。

对应 SDK 调用：
  client.chat.completions.create(model=..., messages=[system, user], ...)

对照 Responses 通路时盯两处：
  - system → messages[0].role == "system"
  - user   → messages[1].role == "user"

"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from scenic_cs_agent.runtime.model_types import (
    ModelAdapter,
    ModelCapabilities,
    ModelRequest,
    ModelResult,
)


class DeepSeekChatAdapter(ModelAdapter):
    name = "deepseek-chat"
    capabilities = ModelCapabilities(
        chat_completions=True,
        responses=False,
        structured_output="json_mode",  # DeepSeek Chat 侧常见是 json_object，不是原生 schema
        tool_calling=True,
        supports_temperature=True,
        supports_top_p=True,
    )

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
    ) -> None:
        self.model = model
        # OpenAI 官方 SDK，只改 base_url + key 就能打 DeepSeek
        self.client = OpenAI(
            api_key=api_key or os.environ["DEEPSEEK_API_KEY"],
            base_url=base_url,
            timeout=60.0,  # 流式可能更久，略放宽
            max_retries=0,  # 重试留给上层 Gateway/Loop，避免 SDK×上层双重重试
        )

    def _messages_and_kwargs(
        self, request: ModelRequest
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """generate / stream 共用：拼 messages 与可选采样参数。"""
        system = request.system
        kwargs: dict[str, Any] = {}

        if request.output_schema:
            system += (
                "\n必须输出 json，并符合此 JSON Schema：\n"
                + json.dumps(request.output_schema, ensure_ascii=False)
            )
            kwargs["response_format"] = {"type": "json_object"}

        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": request.user},
        ]
        return messages, kwargs

    def generate(self, request: ModelRequest) -> ModelResult:
        """非流式：等整段生成完再返回 ModelResult。"""
        messages, kwargs = self._messages_and_kwargs(request)

        raw = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=request.max_output_tokens,
            **kwargs,
        )

        # Chat 返回：choices[0].message.content；usage 字段名是 prompt/completion_tokens
        choice = raw.choices[0]
        text = choice.message.content or ""
        usage = raw.usage
        data = json.loads(text) if request.output_schema else None

        return ModelResult(
            kind="structured" if data is not None else "text",
            text=text,
            data=data,
            finish_reason=choice.finish_reason,
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            request_id=raw.id,
        )
