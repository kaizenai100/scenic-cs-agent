"""测试环境的进程内假模型。

不读密钥，也不访问网络。配置里 kind=fake 的 provider 会用到它，
这样 APP_ENV=test 时不配 Key 也能把一轮对话跑完。

行为是固定的，方便本地和测试对齐：
  - 用户提到了已知景区，并且这轮还没拿到工具结果：调用 get_open_hours；
  - 已经有工具结果：回一段不另编票价的话；
  - 其他情况：请用户补充景区名和日期，不编造营业数据。
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from scenic_cs_agent.runtime.model_types import (
    ModelAdapter,
    ModelCapabilities,
    ModelRequest,
    ModelResult,
    ModelStreamEvent,
)
from scenic_cs_agent.tools.mock_data import SCENICS

# 还没查过资料时的回复。不写具体票价和开放时间。
_NO_TOOL = "可以先告诉我景区名和日期。具体票价和开放时间我需要先查资料，不会直接编。"
# 工具已经回过结果。正文不重复抄数字，避免和工具结果各说各的。
_AFTER_TOOL = "已根据查询结果回答。开放时间和票价以刚才查到的为准，我不会另外编造。"


class FakeModelAdapter(ModelAdapter):
    """固定剧本的适配器。generate 和 stream 说的是同一段话。"""

    name = "fake"
    capabilities = ModelCapabilities(
        chat_completions=True,
        responses=False,
        structured_output="prompt_only",
        tool_calling=True,
        supports_temperature=False,
        supports_top_p=False,
    )

    def generate(self, request: ModelRequest) -> ModelResult:
        if _has_tool_result(request):
            return _text(_AFTER_TOOL)
        scenic = _match_scenic(_user_text(request))
        if scenic is not None and "get_open_hours" in _offered_tools(request):
            arguments = json.dumps({"scenic_name": scenic}, ensure_ascii=False)
            return ModelResult(
                kind="tool_calls",
                text="",
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "id": "call_fake_open_hours",
                        "name": "get_open_hours",
                        "arguments": arguments,
                    }
                ],
                input_tokens=8,
                output_tokens=4,
            )
        return _text(_NO_TOOL)

    def stream(
        self,
        request: ModelRequest,
        *,
        cancel_event: object | None = None,
    ) -> Iterator[ModelStreamEvent]:
        """非 Agent 路径才会走到这里。取消旗标在两条事件之间看一眼。"""

        result = self.generate(request)
        if _cancelled(cancel_event):
            return
        if result.text:
            yield ModelStreamEvent(type="text.delta", data={"delta": result.text})
        if _cancelled(cancel_event):
            return
        yield ModelStreamEvent(
            type="model.finished",
            data={"finish_reason": result.finish_reason or "stop"},
        )
        yield ModelStreamEvent(
            type="model.usage",
            data={
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "total_tokens": result.input_tokens + result.output_tokens,
            },
        )


def _text(text: str) -> ModelResult:
    return ModelResult(
        kind="text",
        text=text,
        finish_reason="stop",
        input_tokens=8,
        output_tokens=len(text),
    )


def _cancelled(cancel_event: object | None) -> bool:
    is_set = getattr(cancel_event, "is_set", None)
    return bool(callable(is_set) and is_set())


def _has_tool_result(request: ModelRequest) -> bool:
    return any(
        isinstance(message, dict) and message.get("role") == "tool"
        for message in (request.messages or [])
    )


def _user_text(request: ModelRequest) -> str:
    parts = [request.user]
    for message in request.messages or []:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


def _offered_tools(request: ModelRequest) -> set[str]:
    names: set[str] = set()
    for tool in request.tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.add(function["name"])
    return names


def _match_scenic(text: str) -> str | None:
    """正文里出现的景区名。多个都出现时，取更靠前的；同一位置取更长的名字。"""

    found: list[tuple[int, str]] = []
    for name in SCENICS:
        index = text.find(name)
        if index >= 0:
            found.append((index, name))
    if not found:
        return None
    found.sort(key=lambda item: (item[0], -len(item[1])))
    return found[0][1]
