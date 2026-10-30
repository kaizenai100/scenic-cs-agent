"""OpenAI · Responses Adapter（选做对照；需要 OPENAI_API_KEY）。

和 DeepSeekResponsesAdapter 结构几乎一样，差别主要是：
  - key 环境变量不同
  - base_url 用 OpenAI 默认（不写 deepseek 域名）
  - 默认模型名不同

日常练习用 DeepSeek 两条通路即可；本文件证明「再换一家也还是 generate()」。
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


class OpenAIResponsesAdapter(ModelAdapter):
    name = "openai-responses"
    capabilities = ModelCapabilities(
        chat_completions=False,
        responses=True,
        structured_output="native_schema",
        tool_calling=True,
        supports_temperature=True,
        supports_top_p=True,
    )

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1-mini",
    ) -> None:
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ["OPENAI_API_KEY"],
            timeout=30.0,
            max_retries=0,
        )

    def generate(self, request: ModelRequest) -> ModelResult:
        kwargs: dict[str, Any] = {}

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
            instructions=request.system,
            input=request.user,
            max_output_tokens=request.max_output_tokens,
            **kwargs,
        )

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
