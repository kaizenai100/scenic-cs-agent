"""步骤 9：MCP 工具经 ToolRuntime 进入 Agent Loop。

远端工具先被收成普通 Tool（名字带服务名前缀），再和本地工具放进同一个 ToolRuntime。
Loop 仍然只调用 runtime.execute：校验、白名单、超时、审计都不绕开。

本示例不需要模型密钥：
  1) 按 yaml 拉起本仓的 stdio mock 服务，脚本多轮走通查单 + 开放时间；
  2) 同一 Runtime 上演示非法参数、未注册工具；
  3) 故意连一个不存在的命令，看启动失败怎么报；
  4) 有 DEEPSEEK_API_KEY 时才打真模型，没有就跳过。

运行：uv run python examples/step09_mcp/main.py
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv

from scenic_cs_agent.config import McpConfig, McpServerConfig, McpToolPolicy, load_config
from scenic_cs_agent.mcp_client import McpHub, McpStartupError
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
# 审计日志打出来，能看到每次工具调用的一行 JSON（参数已脱敏）。
logging.basicConfig(level=logging.WARNING, format="%(message)s")
_audit = logging.getLogger("scenic_cs_agent.tool_audit")
_audit.setLevel(logging.INFO)
_audit_handler = logging.StreamHandler()
_audit_handler.setFormatter(logging.Formatter("%(message)s"))
_audit.addHandler(_audit_handler)
_audit.propagate = False


class ScriptedAdapter(ModelAdapter):
    """按队列吐出预定结果，不消耗模型 API。"""

    name = "scripted-mcp"
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


def _banner(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def _call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"id": call_id, "name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}


def _show(label: str, result) -> None:
    code = result.error_code.value if result.error_code else None
    print(f"- {label}: ok={result.ok} code={code} attempts={result.attempts} → {result.to_model_payload()}")


def demo_scripted_and_governance() -> None:
    """真实 stdio 服务 + 脚本模型。治理演示也打在这条连接上。"""
    _banner("1) stdio mock 服务，经 ToolRuntime 进 Loop")
    cfg = load_config()
    if not cfg.mcp.enabled:
        print(f"APP_ENV={cfg.env} 关闭了 MCP，跳过真实连接。把 mcp.enabled 设为 true 再跑这一段。")
        return

    ctx = ExecutionContext(run_id="demo-mcp", user_id="u-1001", channel="app", scene="presale")
    local_names = [tool.name for tool in SCENIC_TOOLS]
    with McpHub(cfg.mcp, local_names=local_names) as hub:
        if hub.errors:
            print("非必需服务没连上：")
            for item in hub.errors:
                print(f"  {item}")
        print("注册结果：")
        for report in hub.reports:
            if report.state == "registered":
                print(f"  + {report.exposed_name}  ← 远端 {report.server}.{report.remote_name}")
            else:
                print(f"  - {report.server}.{report.remote_name} 跳过（{report.reason}）")

        runtime = ToolRuntime([*SCENIC_TOOLS, *hub.tools])
        print("售前可见：", ", ".join(runtime.names(ctx)))

        adapter = ScriptedAdapter(
            [
                ModelResult(
                    kind="tool_calls",
                    tool_calls=[
                        _call("call_order", "scenic_ticket__query_ticket_order", {"order_id": "O20261101"}),
                        _call("call_hours", "scenic_ticket__get_opening_hours", {"scenic_name": "云栖山"}),
                    ],
                    text="",
                ),
                ModelResult(
                    kind="text",
                    text="订单 O20261101 已支付，云栖山成人票，游玩日 2026-11-08。云栖山 08:00-17:30 开放，周一闭园（法定节假日除外）。",
                    finish_reason="stop",
                ),
            ]
        )
        out = run_agent_loop(
            adapter=adapter,
            runtime=runtime,
            ctx=ctx,
            system="你是景区客服。事实以工具结果为准。",
            user="帮我看下订单 O20261101，顺便说下云栖山几点开门。",
            guard=LoopGuard(max_turns=4),
        )
        print(f"stop_reason={out.stop_reason} turns={out.turns}")
        print(f"tools: {[(item.name, item.ok, item.latency_ms) for item in out.tool_results]}")
        print(f"final: {out.text}")

        _banner("2) 同一 Runtime 的治理（坏参数不会打到 MCP 服务）")
        _show("参数不符合 schema", runtime.execute(
            _call("bad_args", "scenic_ticket__query_ticket_order", {"order_id": "不是订单号"}),
            ctx,
        ))
        _show("未注册的名字", runtime.execute(
            {"id": "missing", "name": "scenic_ticket__drop_database", "arguments": "{}"},
            ctx,
        ))
        _show("订单不存在（服务返回的说明会回到模型）", runtime.execute(
            _call("no_order", "scenic_ticket__query_ticket_order", {"order_id": "O00000000"}),
            ctx,
        ))


def demo_startup_failure() -> None:
    """命令不存在：启动失败要看得见，不能装成「没有工具」悄悄继续。"""
    _banner("3) 服务启动失败")
    broken = McpConfig(
        servers={
            "missing": McpServerConfig(
                transport="stdio",
                command="scenic-mcp-missing-bin",
                startup_timeout_seconds=3,
                tools={"query_ticket_order": McpToolPolicy(read_only=True)},
            )
        }
    )
    try:
        McpHub(broken).connect()
    except McpStartupError as exc:
        print(f"McpStartupError: {exc}")


def demo_live_model() -> None:
    _banner("4) 真模型（可选）")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("未设置 DEEPSEEK_API_KEY，跳过。前面的脚本段已经走过 MCP → ToolRuntime → Loop。")
        return

    cfg = load_config()
    if not cfg.mcp.enabled:
        print("当前环境关闭了 MCP，跳过真模型。")
        return
    ctx = ExecutionContext(run_id="demo-mcp-live", user_id="u-1001", channel="app", scene="presale")
    with McpHub(cfg.mcp, local_names=[tool.name for tool in SCENIC_TOOLS]) as hub:
        runtime = ToolRuntime([*SCENIC_TOOLS, *hub.tools])
        system = render_scenic_presale(allowed_tools=runtime.names(ctx)).text
        out = run_agent_loop(
            adapter=DeepSeekChatAdapter(),
            runtime=runtime,
            ctx=ctx,
            system=system,
            user="订单 O20261101 是什么状态？云栖山的开放时间呢？",
            guard=LoopGuard(max_turns=6),
            temperature=0.2,
        )
        print(f"stop_reason={out.stop_reason} turns={out.turns}")
        print(f"tools: {[(item.name, item.ok) for item in out.tool_results]}")
        print(f"final:\n{out.text}")


if __name__ == "__main__":
    demo_scripted_and_governance()
    demo_startup_failure()
    demo_live_model()
