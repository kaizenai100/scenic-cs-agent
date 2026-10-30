"""DeepSeek · Chat Completions 通路（本仓默认）。

对应 SDK 调用：
  client.chat.completions.create(model=..., messages=[system, user], ...)

对照 Responses 通路时盯两处：
  - system → messages[0].role == "system"
  - user   → messages[1].role == "user"

步骤 3：generate() 仍是一次性整段；
stream() 打开 stream=True，把厂商 chunk 收成 ModelStreamEvent。
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from scenic_cs_agent.runtime.model_types import (
    ModelAdapter,
    ModelCapabilities,
    ModelRequest,
    ModelResult,
    ModelStreamEvent,
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
        data = None
        if request.output_schema:
            # 解析失败不在这里抛：交给上层结构化执行器决定修复/兜底
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = None

        return ModelResult(
            kind="structured" if request.output_schema else "text",
            text=text,
            data=data,
            finish_reason=choice.finish_reason,
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            request_id=raw.id,
        )

    def stream(self, request: ModelRequest) -> Iterator[ModelStreamEvent]:
        """流式：边收边 yield 统一事件。

        注意：
        - stream=True 不会让模型推理更快，价值是「可观察 / 可取消 / 先出字」
        - 这里只做 Adapter 层归一；还没上 FastAPI SSE（步骤稍后）
        - include_usage=True：最后一块常带 usage（具体以厂商为准）
        """
        messages, kwargs = self._messages_and_kwargs(request)

        raw = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=request.max_output_tokens,
            stream=True,
            stream_options={"include_usage": True},
            **kwargs,
        )

        for chunk in raw:
            choice = chunk.choices[0] if chunk.choices else None

            # 增量正文：打字机效果的数据面
            if choice and choice.delta and choice.delta.content:
                yield ModelStreamEvent(
                    type="text.delta",
                    data={"delta": choice.delta.content},
                )

            # 模型侧结束原因（stop / length / ...）
            if choice and choice.finish_reason:
                yield ModelStreamEvent(
                    type="model.finished",
                    data={"finish_reason": choice.finish_reason},
                )

            # 用量常在流末尾单独一块；没有 choices 也正常
            if chunk.usage is not None:
                yield ModelStreamEvent(
                    type="model.usage",
                    data={
                        "input_tokens": chunk.usage.prompt_tokens,
                        "output_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    },
                )
