"""步骤 7：任务内 Agent Loop（手写 while：模型 ↔ ToolRuntime）。

本示例分三段，验收不依赖模型运气：
  1) Scripted 多轮：预定 tool_calls → Runtime → 最终回答（演示协议停止）；
  2) LoopGuard 防御：max_turns / 重复指纹 / 取消（全脚本驱动）；
  3) 真模型（可选）：问拙政园明天票价与开放时间；无 DEEPSEEK_API_KEY 则跳过并提示。

运行：uv run python examples/step07_agent_loop/main.py
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any

from dotenv import load_dotenv

from scenic_cs_agent.prompts import render_scenic_presale
from scenic_cs_agent.runtime.agent_loop import run_agent_loop
from scenic_cs_agent.runtime.deepseek_chat_adapter import DeepSeekChatAdapter
from scenic_cs_agent.runtime.loop_guard import LoopGuard
from scenic_cs_agent.runtime.model_types import (
    ModelAdapter,
    ModelCapabilities,
    ModelRequest,
    ModelResult,
)
from scenic_cs_agent.tools import SCENIC_TOOLS, ExecutionContext, ToolRuntime

load_dotenv()
logging.basicConfig(level=logging.WARNING)  # 示例里看 loop 事件即可，审计默认降噪


class ScriptedAdapter(ModelAdapter):
    """示例用假模型：按队列吐出预定结果，专门测 Loop / Guard，不消耗 API。"""

    name = "scripted-demo"
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

    def generate(self, request: ModelRequest) -> ModelResult:
        if not self._script:
            return ModelResult(kind="text", text="(script exhausted)", finish_reason="stop")
        return self._script.pop(0)


def _tc(call_id: str, name: str, scenic: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "name": name,
        "arguments": json.dumps({"scenic_name": scenic}, ensure_ascii=False),
    }


def _banner(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def demo_scripted_happy_path() -> None:
    """脚本多轮：并行查开放时间+票价，再给出最终回答。"""
    _banner("1) Scripted 多轮（protocol_stop）")
    adapter = ScriptedAdapter(
        [
            ModelResult(
                kind="tool_calls",
                tool_calls=[
                    _tc("call_h", "get_open_hours", "拙政园"),
                    _tc("call_p", "get_ticket_price", "拙政园"),
                ],
                text="",
            ),
            ModelResult(
                kind="text",
                text=(
                    "拙政园明天正常开放：07:30-17:30（17:00 停止入园）；"
                    "成人票 90 元，学生票 45 元。需要联票可以再说。"
                ),
                finish_reason="stop",
            ),
        ]
    )
    ctx = ExecutionContext(run_id="demo-scripted", user_id="u-1001", channel="app", scene="presale")
    runtime = ToolRuntime(SCENIC_TOOLS)
    system = render_scenic_presale(allowed_tools=runtime.names(ctx)).text
    out = run_agent_loop(
        adapter=adapter,
        runtime=runtime,
        ctx=ctx,
        system=system,
        user="明天想去拙政园，几点开门？门票多少钱？",
        guard=LoopGuard(max_turns=6),
    )
    print(f"stop_reason={out.stop_reason} turns={out.turns}")
    print(f"final: {out.text}")
    print(f"tool_results: {[ (r.name, r.ok) for r in out.tool_results ]}")


def demo_guard_cases() -> None:
    """不依赖模型：强制触发三种护栏停止原因。"""
    _banner("2) LoopGuard 防御（脚本强制触发）")
    runtime = ToolRuntime(SCENIC_TOOLS)
    ctx = ExecutionContext(run_id="demo-guard", user_id="u-1001", channel="app", scene="presale")
    system = "你是景区售前客服。"

    # 2a max_turns：每轮都还想调工具，第 2 次 begin_turn 时被拦。
    call = _tc("m1", "get_open_hours", "云栖山")
    out = run_agent_loop(
        adapter=ScriptedAdapter(
            [
                ModelResult(kind="tool_calls", tool_calls=[call], text=""),
                ModelResult(kind="tool_calls", tool_calls=[call], text=""),
            ]
        ),
        runtime=runtime,
        ctx=ctx,
        system=system,
        user="云栖山几点开？",
        guard=LoopGuard(max_turns=1),
    )
    print(f"- max_turns → stop_reason={out.stop_reason} turns={out.turns}")

    # 2b duplicate：第二轮再次点名同一工具+同一参数。
    c1 = _tc("d1", "get_ticket_price", "云栖山")
    c2 = _tc("d2", "get_ticket_price", "云栖山")
    out = run_agent_loop(
        adapter=ScriptedAdapter(
            [
                ModelResult(kind="tool_calls", tool_calls=[c1], text=""),
                ModelResult(kind="tool_calls", tool_calls=[c2], text=""),
            ]
        ),
        runtime=runtime,
        ctx=ctx,
        system=system,
        user="云栖山票价？",
        guard=LoopGuard(max_turns=5),
    )
    print(
        f"- duplicate_action → stop_reason={out.stop_reason} "
        f"executed={len(out.tool_results)}（第二次应未执行）"
    )

    # 2c cancelled：开场即取消。
    ev = threading.Event()
    ev.set()
    out = run_agent_loop(
        adapter=ScriptedAdapter([ModelResult(kind="text", text="nope", finish_reason="stop")]),
        runtime=runtime,
        ctx=ctx,
        system=system,
        user="任意",
        guard=LoopGuard(max_turns=5),
        cancel_event=ev,
    )
    print(f"- cancelled → stop_reason={out.stop_reason} turns={out.turns}")


def demo_live_model() -> None:
    """真模型路径：需要 DEEPSEEK_API_KEY；没有就跳过。"""
    _banner("3) 真模型多轮（可选）")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("未设置 DEEPSEEK_API_KEY，跳过真模型段。脚本/护栏段已足够验收 Loop。")
        return

    runtime = ToolRuntime(SCENIC_TOOLS)
    ctx = ExecutionContext(run_id="demo-live", user_id="u-1001", channel="app", scene="presale")
    system = render_scenic_presale(allowed_tools=runtime.names(ctx)).text
    out = run_agent_loop(
        adapter=DeepSeekChatAdapter(),
        runtime=runtime,
        ctx=ctx,
        system=system,
        user="明天想去拙政园，开放时间是？门票价格呢？",
        guard=LoopGuard(max_turns=6),
        temperature=0.2,
    )
    print(f"stop_reason={out.stop_reason} turns={out.turns}")
    print(f"tools used: {[(r.name, r.ok, r.latency_ms) for r in out.tool_results]}")
    print(f"final:\n{out.text}")


if __name__ == "__main__":
    demo_scripted_happy_path()
    demo_guard_cases()
    demo_live_model()
