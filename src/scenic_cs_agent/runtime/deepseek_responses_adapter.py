"""DeepSeek · Responses API 通路（同一把 DEEPSEEK_API_KEY）。

官方说明：https://api-docs.deepseek.com/guides/responses_api/
常用模型：deepseek-flash、deepseek-v4-pro（以官网为准）。

对应 SDK 调用：
  client.responses.create(model=..., instructions=..., input=..., ...)

和 Chat Completions 的关键差异（验收口述用）：
  1) 方法名：responses.create vs chat.completions.create
  2) system → instructions；user → input（不是 messages 数组）
  3) 文本在 output_text；usage 字段名是 input_tokens / output_tokens

步骤 3：先给 Chat 做 stream()；Responses 流式事件形状不同
（response.output_text.delta 等），下一步再补，避免一次吞两套。
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


class DeepSeekResponsesAdapter(ModelAdapter):
    name = "deepseek-responses"
    capabilities = ModelCapabilities(
        chat_completions=False,
        responses=True,
        structured_output="native_schema",  # Responses 可用 text.format = json_schema
        tool_calling=True,
        supports_temperature=True,
        supports_top_p=True,
    )

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-flash",
        base_url: str = "https://api.deepseek.com",
    ) -> None:
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ["DEEPSEEK_API_KEY"],
            base_url=base_url,
            timeout=60.0,
            max_retries=0,
        )

    def generate(self, request: ModelRequest) -> ModelResult:
        kwargs: dict[str, Any] = {}

        # Responses 原生结构化：走 text.format.json_schema（本步可不测）
        if request.output_schema:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "agent_result",
                    "strict": True,
                    "schema": request.output_schema,
                }
            }

        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p

        raw = self.client.responses.create(
            model=self.model,
            instructions=request.system,  # ← 对应 Chat 的 system message
            input=request.user,  # ← 对应 Chat 的 user message
            max_output_tokens=request.max_output_tokens,
            **kwargs,
        )

        # Responses 用 status 表示是否完成（completed / incomplete / failed）
        if raw.status != "completed":
            raise RuntimeError(f"MODEL_NOT_COMPLETED: {raw.status}")

        text = raw.output_text or ""
        data = json.loads(text) if request.output_schema else None
        usage = raw.usage

        return ModelResult(
            kind="structured" if data is not None else "text",
            text=text,
            data=data,
            finish_reason=raw.status,
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            request_id=raw.id,
        )
