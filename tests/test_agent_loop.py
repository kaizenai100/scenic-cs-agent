"""步骤 7 · Agent Loop 单元测试。

用 ScriptedAdapter 喂预定的 tool_calls / 最终文本，不依赖真实模型，保证：
  - 工具结果只经 ToolRuntime；
  - tool_call id 在 assistant → Runtime → tool 消息链路中保持一致；
  - protocol_stop / max_turns / duplicate_action / cancelled 四种停止原因可断言。
"""

from __future__ import annotations

import json
import threading
from typing import Any

import pytest

from scenic_cs_agent.runtime.agent_loop import run_agent_loop
from scenic_cs_agent.runtime.loop_guard import LoopGuard, fingerprint_action
from scenic_cs_agent.runtime.model_types import (
    ModelAdapter,
    ModelCapabilities,
    ModelRequest,
    ModelResult,
)
from scenic_cs_agent.tools import SCENIC_TOOLS, ExecutionContext, ToolRuntime


class ScriptedAdapter(ModelAdapter):
    """按队列依次返回预定 ModelResult；记录每次收到的 ModelRequest 供断言。"""

    name = "scripted"
    capabilities = ModelCapabilities(
        chat_completions=True,
        responses=False,
        structured_output="prompt_only",
        tool_calling=True,
        supports_temperature=True,
        supports_top_p=True,
    )

    def __init__(self, script: list[ModelResult]) -> None:
        self._script = list(script)
        self.requests: list[ModelRequest] = []

    def generate(self, request: ModelRequest) -> ModelResult:
        self.requests.append(request)
        if not self._script:
            return ModelResult(kind="text", text="(script exhausted)", finish_reason="stop")
        return self._script.pop(0)


def _ctx() -> ExecutionContext:
    return ExecutionContext(run_id="t-loop", user_id="u-test", channel="app", scene="presale")


def _tc(call_id: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"id": call_id, "name": name, "arguments": json.dumps(args, ensure_ascii=False)}


def test_protocol_stop_and_tool_call_id_preserved():
    """多轮：先两个 tool_calls → Runtime → 再最终文本；id 必须贯穿消息。"""
    call_hours = _tc("call_hours_1", "get_open_hours", {"scenic_name": "拙政园"})
    call_price = _tc("call_price_1", "get_ticket_price", {"scenic_name": "拙政园"})
    adapter = ScriptedAdapter(
        [
            ModelResult(kind="tool_calls", tool_calls=[call_hours, call_price], text=""),
            ModelResult(
                kind="text",
                text="拙政园明天 07:30-17:30 开放，成人票 90 元。",
                finish_reason="stop",
            ),
        ]
    )
    events: list[tuple[str, dict]] = []
    runtime = ToolRuntime(SCENIC_TOOLS)
    result = run_agent_loop(
        adapter=adapter,
        runtime=runtime,
        ctx=_ctx(),
        system="你是景区售前客服。",
        user="拙政园明天几点开门？门票多少钱？",
        guard=LoopGuard(max_turns=5),
        on_event=lambda e, p: events.append((e, p)),
    )

    assert result.stop_reason == "protocol_stop"
    assert result.text and "拙政园" in result.text
    assert result.turns == 2
    assert len(result.tool_results) == 2
    assert all(tr.ok for tr in result.tool_results)

    # 第二轮请求必须已带上 role=tool，且 tool_call_id 与模型原始 id 一致。
    assert len(adapter.requests) == 2
    second_msgs = adapter.requests[1].messages or []
    tool_msgs = [m for m in second_msgs if m.get("role") == "tool"]
    assert {m["tool_call_id"] for m in tool_msgs} == {"call_hours_1", "call_price_1"}

    # assistant 消息里的 tool_calls.id 同样保留。
    assistant_with_tools = [
        m for m in result.messages if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    assert len(assistant_with_tools) == 1
    ids = {tc["id"] for tc in assistant_with_tools[0]["tool_calls"]}
    assert ids == {"call_hours_1", "call_price_1"}

    # 可观测事件至少包含 turn_start / tool_exec / 最终 turn_end。
    kinds = [e for e, _ in events]
    assert "turn_start" in kinds and "tool_exec" in kinds and "turn_end" in kinds


def test_results_go_through_tool_runtime_not_direct_func():
    """Loop 不得绕过 Runtime：故意点名不可见/非法工具，应拿到 Runtime 错误码。"""
    bad = _tc("call_bad", "create_refund_request", {
        "order_id": "O20261101",
        "phone": "13812345678",
        "reason": "test",
    })
    adapter = ScriptedAdapter(
        [
            ModelResult(kind="tool_calls", tool_calls=[bad], text=""),
            ModelResult(kind="text", text="抱歉，售前场景无法退票。", finish_reason="stop"),
        ]
    )
    result = run_agent_loop(
        adapter=adapter,
        runtime=ToolRuntime(SCENIC_TOOLS),
        ctx=_ctx(),  # scene=presale → 退票工具不可见
        system="sys",
        user="帮我退票",
        guard=LoopGuard(max_turns=5),
        on_event=lambda *_: None,
    )
    assert result.stop_reason == "protocol_stop"
    assert len(result.tool_results) == 1
    assert result.tool_results[0].ok is False
    assert result.tool_results[0].error_code is not None
    # 回灌给模型的 tool 消息应带上 ok=false，而不是静默吞掉。
    tool_msg = next(m for m in result.messages if m.get("role") == "tool")
    payload = json.loads(tool_msg["content"])
    assert payload["ok"] is False
    assert tool_msg["tool_call_id"] == "call_bad"


def test_max_turns_stops_when_model_keeps_calling_tools():
    """max_turns=1：第一轮允许 tool_calls；若还要再调模型，第二轮 begin_turn 触发 max_turns。"""
    call = _tc("c1", "get_open_hours", {"scenic_name": "云栖山"})
    # 脚本只准备「永远返回 tool_calls」；第二轮根本不该再 generate。
    adapter = ScriptedAdapter(
        [
            ModelResult(kind="tool_calls", tool_calls=[call], text=""),
            ModelResult(kind="tool_calls", tool_calls=[call], text=""),  # 不应被消费
        ]
    )
    result = run_agent_loop(
        adapter=adapter,
        runtime=ToolRuntime(SCENIC_TOOLS),
        ctx=_ctx(),
        system="sys",
        user="云栖山几点开？",
        guard=LoopGuard(max_turns=1),
        on_event=lambda *_: None,
    )
    assert result.stop_reason == "max_turns"
    assert result.turns == 2  # begin_turn 第二次时 turns_used 变为 2 > max
    assert len(adapter.requests) == 1  # 只真正调了一次模型
    assert len(result.tool_results) == 1


def test_duplicate_action_stops():
    """同一工具+相同参数出现第二次 → duplicate_action，不再执行第二次。"""
    call = _tc("c1", "get_ticket_price", {"scenic_name": "云栖山"})
    # 第二轮再次点名完全相同的动作（指纹相同）。
    call2 = _tc("c2", "get_ticket_price", {"scenic_name": "云栖山"})
    adapter = ScriptedAdapter(
        [
            ModelResult(kind="tool_calls", tool_calls=[call], text=""),
            ModelResult(kind="tool_calls", tool_calls=[call2], text=""),
        ]
    )
    result = run_agent_loop(
        adapter=adapter,
        runtime=ToolRuntime(SCENIC_TOOLS),
        ctx=_ctx(),
        system="sys",
        user="云栖山票价？",
        guard=LoopGuard(max_turns=5),
        on_event=lambda *_: None,
    )
    assert result.stop_reason == "duplicate_action"
    assert len(result.tool_results) == 1  # 第二次未执行
    # 指纹归一：键序不同也应相同。
    assert fingerprint_action("get_ticket_price", '{"scenic_name":"云栖山"}') == fingerprint_action(
        "get_ticket_price", {"scenic_name": "云栖山"}
    )


def test_cancelled_stops_before_model_call():
    """cancel_event 已 set：第一轮开头即 cancelled，不调模型。"""
    adapter = ScriptedAdapter(
        [ModelResult(kind="text", text="不该出现", finish_reason="stop")]
    )
    ev = threading.Event()
    ev.set()
    result = run_agent_loop(
        adapter=adapter,
        runtime=ToolRuntime(SCENIC_TOOLS),
        ctx=_ctx(),
        system="sys",
        user="hi",
        guard=LoopGuard(max_turns=5),
        cancel_event=ev,
        on_event=lambda *_: None,
    )
    assert result.stop_reason == "cancelled"
    assert adapter.requests == []
    assert result.turns == 0


def test_fingerprint_normalizes_key_order():
    a = fingerprint_action("t", '{"b":2,"a":1}')
    b = fingerprint_action("t", '{"a":1,"b":2}')
    assert a == b
