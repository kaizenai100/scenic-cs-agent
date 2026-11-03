"""结构化输出的可选修复重试。

模型和步骤 5 的 classify_intent 是同一类做法：校验失败就把错误附在下一次请求里，让模型改。
这里做成 Gateway 上的可选钩子，有两个边界：

  1. 不替换 structured/runner.py。意图识别的兜底话术、槽位契约仍由那一层负责。
  2. Gateway 只负责「再调用一次」。校验函数由调用方传入，网关不绑定某一个 Pydantic 模型。

修复次数和传输层重试是两笔账。传输重试发生在还没拿到完整结果时；
修复重试发生在已经拿到文本、但校验没过时。混在一起会把 429 和「JSON 缺字段」用同一套次数算完。
"""

from __future__ import annotations

from dataclasses import replace

from scenic_cs_agent.runtime.model_types import ModelRequest


def with_repair_feedback(request: ModelRequest, *, raw_text: str | None, error: str) -> ModelRequest:
    """基于原始请求附上最近一次错误。

    每次都从最初的 request 出发，只带最新一次校验错误，避免修复轮次把提示词越拼越长。
    messages 非空时必须改 messages：适配器会优先发完整历史，只改 user 字段不会进模型。
    """
    feedback = (
        "[上次输出不合法，请修正后只输出要求的 JSON]\n"
        f"上次输出：{raw_text}\n"
        f"校验错误：{error[:800]}"
    )
    if request.messages is not None:
        messages = [dict(item) for item in request.messages]
        messages.append({"role": "user", "content": feedback})
        return replace(request, messages=messages)
    return replace(request, user=f"{request.user}\n\n{feedback}")
