"""任务内 Agent Loop（手写，不用 LangGraph / 外部 Agent 框架）。

本模块把「单轮工具」升级成「多轮：模型 ↔ 工具」循环。范围刻意收窄：
  - 只管一次用户任务内部的 while 循环（task-inner）；
  - 不负责任务调度、不写 Checkpoint、不接 FastAPI SSE（后续步骤再挂）。

四层对照（景区客服语境，不是代码仓库 Agent）：
  Function Calling → 模型返回 tool_calls（Adapter 已归一为 {id,name,arguments}）
  Tool Runtime     → 唯一允许执行工具的入口（白名单/校验/审批/超时）
  Agent Loop       → 组装消息 → 调模型 → 若有 tool_calls 则经 Runtime 执行并回灌 → 重复
  LoopGuard        → max_turns / 重复动作指纹 / cancel_event

协议停止 ≠ 业务完成：
  模型本轮不返回 tool_calls（protocol_stop）只说明「协议上它想说话了」；
  并不保证票价/开放时间已经查全、用户问题已经答完。
  真正的「业务完成契约」（Completion Contract）本步只留钩子注释，下步再接。
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from scenic_cs_agent.runtime.loop_guard import LoopGuard
from scenic_cs_agent.runtime.model_types import ModelAdapter, ModelRequest, ModelResult
from scenic_cs_agent.tools.base import ToolResult
from scenic_cs_agent.tools.context import ExecutionContext
from scenic_cs_agent.tools.runtime import ToolRuntime

# 停止原因：给上层（日志 / SSE / 测试断言）看，不要靠猜。
StopReason = Literal[
    "protocol_stop",  # 模型未再请求工具（协议层停止）
    "max_turns",  # 护栏：超过最大轮数
    "duplicate_action",  # 护栏：重复相同工具+参数
    "cancelled",  # 护栏：外部取消
]

# 可观测事件回调：turn_start / tool_exec / turn_end；默认 print，测试可注入收集器。
TurnEventCallback = Callable[[str, dict[str, Any]], None]


def _default_on_event(event: str, payload: dict[str, Any]) -> None:
    """简单结构化日志：示例与本地调试看得见每一轮在干什么。"""
    print(f"[loop] {event} {json.dumps(payload, ensure_ascii=False)}")


@dataclass
class AgentLoopResult:
    """一次任务内循环的收口结果。"""

    # 最终给用户看的文本（protocol_stop 时通常有值；护栏打断时可能是半成品或空）。
    text: str | None
    # 为什么停下来——验收与排障的第一证据。
    stop_reason: StopReason
    # 完整消息历史（含 system / user / assistant+tool_calls / tool），便于断言与回放。
    messages: list[dict[str, Any]] = field(default_factory=list)
    # 实际跑了几轮 generate（与 LoopGuard.turns_used 对齐）。
    turns: int = 0
    # 本任务内所有经 Runtime 执行过的 ToolResult（按时间序）。
    tool_results: list[ToolResult] = field(default_factory=list)


def _assistant_message(result: ModelResult) -> dict[str, Any]:
    """把 ModelResult 收成 OpenAI chat 的 assistant 消息。

    关键：tool_calls 必须保留模型给的 id，后续 role=tool 的 tool_call_id 靠它对齐；
    丢 id = 模型无法把「哪个结果」对应到「哪次调用」。
    """
    msg: dict[str, Any] = {
        "role": "assistant",
        # 有工具调用时 content 常为空；协议允许 null，这里用空串更利于 JSON 序列化。
        "content": result.text or "",
    }
    if result.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    # arguments 保持原始 JSON 字符串；不要在 Loop 里提前 loads，
                    # 解析失败要交给 ToolRuntime 统一返回 INVALID_ARGS。
                    "arguments": tc.get("arguments") or "{}",
                },
            }
            for tc in result.tool_calls
        ]
    return msg


def _tool_message(tr: ToolResult) -> dict[str, Any]:
    """ToolResult → OpenAI role=tool 消息；tool_call_id 必须与请求侧 id 一致。"""
    return {
        "role": "tool",
        "tool_call_id": tr.tool_call_id,
        # content 必须是字符串；把 to_model_payload() 序列化后回灌，模型只看到业务需要的字段。
        "content": json.dumps(tr.to_model_payload(), ensure_ascii=False),
    }


def run_agent_loop(
    *,
    adapter: ModelAdapter,
    runtime: ToolRuntime,
    ctx: ExecutionContext,
    system: str,
    user: str,
    guard: LoopGuard | None = None,
    cancel_event: threading.Event | None = None,
    on_event: TurnEventCallback | None = None,
    max_output_tokens: int = 1024,
    temperature: float | None = None,
) -> AgentLoopResult:
    """手写任务内 Agent Loop：组装消息 → 调模型 →（可选）经 Runtime 执行工具 → 回灌。

    硬约束：
      - 工具只能通过 runtime.execute 执行，禁止在本函数里直接调 Tool.func；
      - 模型给的 arguments 视为不可信输入，校验/白名单全在 Runtime；
      - 本函数不做外层调度、不落盘 Checkpoint。
    """
    guard = guard or LoopGuard()
    emit = on_event or _default_on_event

    # 初始消息：system + 本轮用户问题。后续 assistant / tool 都追加到同一列表。
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    tool_results: list[ToolResult] = []

    while True:
        # ---- 护栏：取消 / 轮数（在调模型之前检查）----
        if reason := guard.check_cancel(cancel_event):
            emit("turn_end", {"stop_reason": reason, "turns": guard.turns_used})
            return AgentLoopResult(
                text=None, stop_reason=reason, messages=messages,
                turns=guard.turns_used, tool_results=tool_results,
            )
        if reason := guard.begin_turn():
            emit("turn_end", {"stop_reason": reason, "turns": guard.turns_used})
            return AgentLoopResult(
                text=None, stop_reason=reason, messages=messages,
                turns=guard.turns_used, tool_results=tool_results,
            )

        emit("turn_start", {"turn": guard.turns_used, "message_count": len(messages)})

        # 可见工具随场景变化；每轮重新 schemas，避免场景切换后仍挂旧清单。
        tools = runtime.schemas(ctx)
        request = ModelRequest(
            # system/user 仍填：兼容只认这两字段的旧路径；真正发给模型的是 messages。
            system=system,
            user=user,
            messages=messages,
            tools=tools or None,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        # 本步用非流式 generate；流式 + SSE 挂接留给后续步骤。
        result = adapter.generate(request)

        # 先把 assistant 消息推进历史（无论是否带 tool_calls），保证回放完整。
        messages.append(_assistant_message(result))

        # ---- 协议停止：模型不再点名工具 ----
        if not result.tool_calls:
            # Completion Contract 钩子（本步不实现）：
            # 这里可以插入「业务完成检查」——例如是否已查到票价且回复覆盖了用户槽位。
            # 协议停止只说明没 tool_calls，不等于业务完成；未完成时可改写 messages 再 continue。
            emit(
                "turn_end",
                {
                    "turn": guard.turns_used,
                    "stop_reason": "protocol_stop",
                    "has_text": bool(result.text),
                },
            )
            return AgentLoopResult(
                text=result.text,
                stop_reason="protocol_stop",
                messages=messages,
                turns=guard.turns_used,
                tool_results=tool_results,
            )

        # ---- 有 tool_calls：逐个经 Runtime 执行，并检查重复指纹 ----
        for call in result.tool_calls:
            name = call["name"]
            arguments = call.get("arguments") or "{}"

            if guard.should_stop_duplicate(name, arguments):
                emit(
                    "turn_end",
                    {
                        "turn": guard.turns_used,
                        "stop_reason": "duplicate_action",
                        "tool": name,
                    },
                )
                return AgentLoopResult(
                    text=result.text,
                    stop_reason="duplicate_action",
                    messages=messages,
                    turns=guard.turns_used,
                    tool_results=tool_results,
                )

            fp = guard.record_action(name, arguments)
            emit(
                "tool_exec",
                {
                    "turn": guard.turns_used,
                    "tool": name,
                    "call_id": call["id"],
                    "fingerprint": fp,
                },
            )
            # 唯一执行入口：模型 args 不可信，全部交给 ToolRuntime。
            tr = runtime.execute(
                {"id": call["id"], "name": name, "arguments": arguments},
                ctx,
            )
            tool_results.append(tr)
            messages.append(_tool_message(tr))

        emit(
            "turn_end",
            {
                "turn": guard.turns_used,
                "tool_count": len(result.tool_calls),
                "continue": True,
            },
        )
        # 工具结果已回灌 → while 继续，让模型基于新消息再生成一轮。


# 薄封装：与函数式入口等价，方便以后挂状态 / 依赖注入。
class AgentLoop:
    """面向对象外壳；内部直接委托 run_agent_loop。"""

    def __init__(
        self,
        *,
        adapter: ModelAdapter,
        runtime: ToolRuntime,
        guard: LoopGuard | None = None,
        on_event: TurnEventCallback | None = None,
    ) -> None:
        self.adapter = adapter
        self.runtime = runtime
        self.guard = guard or LoopGuard()
        self.on_event = on_event

    def run(
        self,
        *,
        ctx: ExecutionContext,
        system: str,
        user: str,
        cancel_event: threading.Event | None = None,
        max_output_tokens: int = 1024,
        temperature: float | None = None,
    ) -> AgentLoopResult:
        return run_agent_loop(
            adapter=self.adapter,
            runtime=self.runtime,
            ctx=ctx,
            system=system,
            user=user,
            guard=self.guard,
            cancel_event=cancel_event,
            on_event=self.on_event,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
