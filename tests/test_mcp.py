"""MCP 工具必须经过 ToolRuntime。

失败场景用假会话，避免真子进程时好时坏。
stdio / streamable HTTP 各留一条真链路，确认注册、调用、关闭是通的。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from scenic_cs_agent.config import McpConfig, McpServerConfig, McpToolPolicy, load_config
from scenic_cs_agent.mcp_client import McpHub, McpStartupError, stdio_parameters
from scenic_cs_agent.mcp_server import server as mcp_http_server
from scenic_cs_agent.runtime.agent_loop import run_agent_loop
from scenic_cs_agent.runtime.loop_guard import LoopGuard
from scenic_cs_agent.runtime.model_types import (
    ModelAdapter,
    ModelCapabilities,
    ModelRequest,
    ModelResult,
)
from scenic_cs_agent.tools import SCENIC_TOOLS, ExecutionContext, Tool, ToolErrorCode, ToolRuntime
from scenic_cs_agent.tools.errors import ToolFailed

ORDER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "order_id": {"type": "string", "pattern": r"^O\d{8}$"},
        "phone": {"type": "string"},
    },
    "required": ["order_id"],
}


def _remote(name: str, schema: dict[str, Any] | None = None, description: str = "查订单") -> SimpleNamespace:
    return SimpleNamespace(name=name, description=description, input_schema=schema or ORDER_SCHEMA)


def _policy(**overrides: Any) -> McpToolPolicy:
    data: dict[str, Any] = {"read_only": True, "timeout_seconds": 3}
    data.update(overrides)
    return McpToolPolicy.model_validate(data)


def _config(tools: dict[str, McpToolPolicy] | None = None, **server: Any) -> McpConfig:
    payload: dict[str, Any] = {
        "transport": "stdio",
        "command": "python",
        "startup_timeout_seconds": 2,
        "required": True,
        "tools": tools or {"query_ticket_order": _policy()},
    }
    payload.update(server)
    return McpConfig(servers={"scenic_ticket": McpServerConfig.model_validate(payload)})


class FakeSession:
    """假的 MCP 会话：能进异步上下文，列出工具，按设定调用或失败。"""

    def __init__(
        self,
        tools: list[Any],
        *,
        enter_error: BaseException | None = None,
        list_error: BaseException | None = None,
        list_sleep: float = 0,
        call_error: BaseException | None = None,
        call_sleep: float = 0,
        result: Any = None,
    ) -> None:
        self.tools = tools
        self.enter_error = enter_error
        self.list_error = list_error
        self.list_sleep = list_sleep
        self.call_error = call_error
        self.call_sleep = call_sleep
        self.result = result
        self.calls: list[dict[str, Any]] = []
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> FakeSession:
        if self.enter_error:
            raise self.enter_error
        self.entered = True
        return self

    async def __aexit__(self, *exc: object) -> bool:
        self.exited = True
        return False

    async def list_tools(self, cursor: str | None = None, **kwargs: Any) -> SimpleNamespace:
        if self.list_sleep:
            await asyncio.sleep(self.list_sleep)
        if self.list_error:
            raise self.list_error
        return SimpleNamespace(tools=self.tools, next_cursor=None)

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        self.calls.append({"name": name, "arguments": arguments or {}, **kwargs})
        if self.call_sleep:
            await asyncio.sleep(self.call_sleep)
        if self.call_error:
            raise self.call_error
        if callable(self.result):
            return self.result(name, arguments)
        if self.result is not None:
            return self.result
        return SimpleNamespace(is_error=False, structured_content={"echo": arguments or {}}, content=[])


def _ctx(scene: str = "presale") -> ExecutionContext:
    return ExecutionContext(run_id="run-mcp", user_id="u-1", channel="app", scene=scene)


def _execute(runtime: ToolRuntime, name: str, arguments: dict[str, Any] | str, ctx: ExecutionContext | None = None, **kwargs: Any):
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return runtime.execute({"id": "c1", "name": name, "arguments": arguments}, ctx or _ctx(), **kwargs)


def _hub(session: FakeSession, config: McpConfig | None = None, **kwargs: Any) -> McpHub:
    return McpHub(config or _config(), client_factory=lambda spec: session, **kwargs)


class ScriptedAdapter(ModelAdapter):
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


def test_config_loads_per_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SCENIC_MCP_TOKEN", "tok-from-env")
    test_cfg = load_config("test")
    prod_cfg = load_config("production")

    assert test_cfg.mcp.enabled is True
    scenic = test_cfg.mcp.servers["scenic_ticket"]
    assert scenic.enabled is True
    assert scenic.transport == "stdio"
    assert scenic.command == "python"
    assert scenic.args == ["-m", "scenic_cs_agent.mcp_server"]
    assert scenic.startup_timeout_seconds == 10
    assert scenic.required is True
    assert scenic.env["SCENIC_MCP_TOKEN"] == "tok-from-env"
    assert set(scenic.tools) == {"query_ticket_order", "get_opening_hours"}
    assert scenic.tools["query_ticket_order"].read_only is True
    assert test_cfg.mcp.servers["ticket_http"].enabled is False
    assert test_cfg.mcp.servers["ticket_http"].transport == "http"

    assert prod_cfg.mcp.enabled is False
    assert prod_cfg.mcp.servers["scenic_ticket"].enabled is False
    # 生产只关开关，公共配置里的超时还在。
    assert prod_cfg.mcp.servers["scenic_ticket"].startup_timeout_seconds == 15


def test_config_rejects_bad_server_shape():
    with pytest.raises(ValidationError):
        McpServerConfig.model_validate({"transport": "stdio", "tools": {"query_ticket_order": {}}})
    with pytest.raises(ValidationError):
        McpServerConfig.model_validate({"transport": "http", "tools": {"query_ticket_order": {}}})
    with pytest.raises(ValidationError):
        McpServerConfig.model_validate({
            "transport": "stdio",
            "command": "python",
            "url": "http://127.0.0.1/mcp",
            "tools": {"query_ticket_order": {}},
        })
    with pytest.raises(ValidationError):
        McpConfig.model_validate({"name_separator": " ", "servers": {}})
    with pytest.raises(ValidationError):
        McpConfig.model_validate({
            "servers": {
                "bad-name": {
                    "transport": "stdio",
                    "command": "python",
                    "tools": {"query_ticket_order": {}},
                }
            }
        })


def test_stdio_parameters_drop_empty_secrets():
    spec = McpServerConfig(
        transport="stdio",
        command="python",
        args=["-m", "scenic_cs_agent.mcp_server"],
        env={"SCENIC_MCP_TOKEN": None, "KEEP": "yes"},
        tools={"query_ticket_order": _policy()},
    )
    params = stdio_parameters(spec)
    assert params.command == "python"
    assert params.args == ["-m", "scenic_cs_agent.mcp_server"]
    assert params.env == {"KEEP": "yes"}

    empty = spec.model_copy(update={"env": {"SCENIC_MCP_TOKEN": None}})
    assert stdio_parameters(empty).env is None


def test_schema_validation_blocks_before_the_server_is_called():
    session = FakeSession([_remote("query_ticket_order")])
    with _hub(session) as hub:
        runtime = ToolRuntime(hub.tools)
        bad_pattern = _execute(runtime, "scenic_ticket__query_ticket_order", {"order_id": "nope"})
        missing = _execute(runtime, "scenic_ticket__query_ticket_order", {})
        broken = _execute(runtime, "scenic_ticket__query_ticket_order", "{")
    assert bad_pattern.error_code is ToolErrorCode.INVALID_ARGS
    assert missing.error_code is ToolErrorCode.INVALID_ARGS
    assert broken.error_code is ToolErrorCode.INVALID_ARGS
    assert session.calls == []


def test_success_uses_remote_name_and_passes_arguments():
    session = FakeSession([_remote("query_ticket_order")])
    with _hub(session) as hub:
        runtime = ToolRuntime(hub.tools)
        result = _execute(runtime, "scenic_ticket__query_ticket_order", {"order_id": "O20261101"})
    assert result.ok
    assert result.content["echo"]["order_id"] == "O20261101"
    assert session.calls[0]["name"] == "query_ticket_order"
    assert "scenic_ticket__" not in session.calls[0]["name"]


def test_text_json_result_is_unpacked_when_structured_content_is_missing():
    session = FakeSession(
        [_remote("query_ticket_order")],
        result=SimpleNamespace(
            is_error=False,
            structured_content=None,
            content=[SimpleNamespace(text='{"order_id": "O20261101", "status": "已支付"}')],
        ),
    )
    with _hub(session) as hub:
        result = _execute(ToolRuntime(hub.tools), "scenic_ticket__query_ticket_order", {"order_id": "O20261101"})
    assert result.ok
    assert result.content["status"] == "已支付"


def test_remote_tool_error_message_reaches_the_model():
    session = FakeSession(
        [_remote("query_ticket_order")],
        result=SimpleNamespace(
            is_error=True,
            structured_content=None,
            content=[SimpleNamespace(text="Error executing tool query_ticket_order: 没有订单 O00000000")],
        ),
    )
    with _hub(session) as hub:
        result = _execute(ToolRuntime(hub.tools), "scenic_ticket__query_ticket_order", {"order_id": "O00000000"})
    assert result.ok is False
    assert result.error_code is ToolErrorCode.EXECUTION_ERROR
    assert result.retryable is False
    assert "没有订单 O00000000" in result.content["error"]
    assert "Error executing tool" not in result.content["error"]


def test_allowlist_hides_tools_the_server_listed():
    session = FakeSession([
        _remote("query_ticket_order"),
        _remote("drop_database", description="别注册我"),
    ])
    with _hub(session) as hub:
        runtime = ToolRuntime(hub.tools)
        names = runtime.names(_ctx())
        hidden = _execute(runtime, "scenic_ticket__drop_database", {})
    assert names == ("scenic_ticket__query_ticket_order",)
    assert hidden.error_code is ToolErrorCode.UNKNOWN_TOOL
    assert session.calls == []
    skipped = [item for item in hub.reports if item.remote_name == "drop_database"]
    assert skipped and skipped[0].reason == "not_allowlisted"


def test_scene_permission_is_enforced_by_runtime():
    session = FakeSession([_remote("query_ticket_order")])
    config = _config(tools={"query_ticket_order": _policy(scenes=["aftersale"])})
    with _hub(session, config) as hub:
        runtime = ToolRuntime(hub.tools)
        denied = _execute(runtime, "scenic_ticket__query_ticket_order", {"order_id": "O20261101"}, _ctx("presale"))
        allowed = _execute(runtime, "scenic_ticket__query_ticket_order", {"order_id": "O20261101"}, _ctx("aftersale"))
    assert denied.error_code is ToolErrorCode.UNKNOWN_TOOL
    assert allowed.ok
    assert len(session.calls) == 1


def test_write_tool_needs_approval_before_the_server_is_called():
    session = FakeSession([_remote("query_ticket_order")])
    config = _config(tools={"query_ticket_order": _policy(read_only=False)})
    with _hub(session, config) as hub:
        runtime = ToolRuntime(hub.tools)
        pending = _execute(runtime, "scenic_ticket__query_ticket_order", {"order_id": "O20261101"})
        assert pending.error_code is ToolErrorCode.APPROVAL_REQUIRED
        assert session.calls == []
        done = _execute(
            runtime, "scenic_ticket__query_ticket_order", {"order_id": "O20261101"}, approved=True
        )
    assert done.ok
    assert len(session.calls) == 1


def test_audit_log_masks_sensitive_fields(caplog: pytest.LogCaptureFixture):
    session = FakeSession([_remote("query_ticket_order")])
    config = _config(tools={"query_ticket_order": _policy(sensitive_fields=["phone"])})
    caplog.set_level(logging.INFO, logger="scenic_cs_agent.tool_audit")
    with _hub(session, config) as hub:
        _execute(
            ToolRuntime(hub.tools),
            "scenic_ticket__query_ticket_order",
            {"order_id": "O20261101", "phone": "13812345678"},
        )
    assert "13812345678" not in caplog.text
    assert "138****78" in caplog.text
    # 打给远端的参数不打码，打码只发生在审计和审批预览。
    assert session.calls[0]["arguments"]["phone"] == "13812345678"


def test_read_only_timeout_is_classified_by_runtime():
    session = FakeSession([_remote("query_ticket_order")], call_sleep=30)
    config = _config(tools={"query_ticket_order": _policy(timeout_seconds=0.2)})
    with _hub(session, config) as hub:
        result = _execute(ToolRuntime(hub.tools), "scenic_ticket__query_ticket_order", {"order_id": "O20261101"})
    assert result.error_code is ToolErrorCode.TIMEOUT
    assert result.attempts == 2
    assert len(session.calls) == 2


def test_write_timeout_is_outcome_unknown_and_not_retried():
    session = FakeSession([_remote("query_ticket_order")], call_sleep=30)
    config = _config(tools={
        "query_ticket_order": _policy(read_only=False, requires_approval=False, timeout_seconds=0.2),
    })
    with _hub(session, config) as hub:
        result = _execute(ToolRuntime(hub.tools), "scenic_ticket__query_ticket_order", {"order_id": "O20261101"})
    assert result.error_code is ToolErrorCode.OUTCOME_UNKNOWN
    assert result.attempts == 1
    assert len(session.calls) == 1


def test_server_crash_mid_call_is_execution_error():
    session = FakeSession([_remote("query_ticket_order")], call_error=ConnectionResetError("reset"))
    with _hub(session) as hub:
        result = _execute(ToolRuntime(hub.tools), "scenic_ticket__query_ticket_order", {"order_id": "O20261101"})
    assert result.error_code is ToolErrorCode.EXECUTION_ERROR
    assert result.attempts == 1
    assert "断开" in result.content["error"]


def test_exception_group_from_a_dead_pipe_is_a_crash():
    session = FakeSession(
        [_remote("query_ticket_order")],
        call_error=ExceptionGroup("stdio", [ConnectionResetError("reset")]),
    )
    with _hub(session) as hub:
        result = _execute(ToolRuntime(hub.tools), "scenic_ticket__query_ticket_order", {"order_id": "O20261101"})
    assert "断开" in result.content["error"]


def test_unexpected_exception_does_not_leak_its_text():
    session = FakeSession(
        [_remote("query_ticket_order")],
        call_error=RuntimeError("token=super-secret"),
    )
    with _hub(session) as hub:
        result = _execute(ToolRuntime(hub.tools), "scenic_ticket__query_ticket_order", {"order_id": "O20261101"})
    assert result.error_code is ToolErrorCode.EXECUTION_ERROR
    assert "RuntimeError" in result.content["error"]
    assert "super-secret" not in result.content["error"]


def test_tool_failed_message_is_visible_and_other_exceptions_stay_typed():
    """Runtime 只对 ToolFailed 放行说明文字，普通异常仍然只留类型名。"""

    class NoArgs(BaseModel):
        pass

    def fail(args: BaseModel, ctx: ExecutionContext) -> dict[str, Any]:
        raise ToolFailed("没有这张订单")

    def explode(args: BaseModel, ctx: ExecutionContext) -> dict[str, Any]:
        raise RuntimeError("token=super-secret")

    runtime = ToolRuntime([
        Tool("fail", "失败", NoArgs, fail),
        Tool("explode", "炸了", NoArgs, explode),
    ])
    failed = _execute(runtime, "fail", {})
    exploded = _execute(runtime, "explode", {})
    assert failed.content["error"] == "没有这张订单"
    assert "super-secret" not in exploded.content["error"]
    assert "RuntimeError" in exploded.content["error"]


def test_name_collision_with_a_local_tool_refuses_the_remote_one():
    session = FakeSession([_remote("query_ticket_order")])
    with pytest.raises(McpStartupError, match="重名"):
        _hub(session, local_names={"scenic_ticket__query_ticket_order"}).connect()
    assert session.exited


def test_optional_collision_skips_one_tool_and_keeps_the_other():
    session = FakeSession([_remote("query_ticket_order"), _remote("get_opening_hours")])
    config = _config(
        tools={
            "query_ticket_order": _policy(),
            "get_opening_hours": _policy(),
        },
        required=False,
    )
    with _hub(session, config, local_names={"scenic_ticket__query_ticket_order"}) as hub:
        names = {tool.name for tool in hub.tools}
    assert names == {"scenic_ticket__get_opening_hours"}
    reasons = {item.remote_name: item.reason for item in hub.reports}
    assert reasons["query_ticket_order"] == "name_collision"
    assert reasons["get_opening_hours"] is None


def test_two_servers_cannot_publish_the_same_exposed_name():
    def factory(spec: McpServerConfig) -> FakeSession:
        tool = "b_c" if spec.command == "first" else "c"
        return FakeSession([_remote(tool)])

    config = McpConfig(
        name_separator="_",
        servers={
            "a": McpServerConfig(
                transport="stdio", command="first", tools={"b_c": _policy()},
            ),
            "a_b": McpServerConfig(
                transport="stdio", command="second", tools={"c": _policy()},
            ),
        },
    )
    with pytest.raises(McpStartupError, match="a_b_c"):
        McpHub(config, client_factory=factory).connect()


def test_required_startup_failure_closes_servers_that_already_connected():
    good = FakeSession([_remote("query_ticket_order")])
    bad = FakeSession([_remote("query_ticket_order")], list_error=ConnectionError("list failed"))

    def factory(spec: McpServerConfig) -> FakeSession:
        return good if spec.command == "good" else bad

    config = McpConfig(servers={
        "ok": McpServerConfig(
            transport="stdio", command="good", tools={"query_ticket_order": _policy()},
        ),
        "bad": McpServerConfig(
            transport="stdio", command="bad", tools={"query_ticket_order": _policy()},
        ),
    })
    with pytest.raises(McpStartupError, match="bad"):
        McpHub(config, client_factory=factory).connect()
    assert good.exited
    assert bad.exited


def test_optional_server_failure_is_recorded_and_the_other_server_stays():
    good = FakeSession([_remote("query_ticket_order")])
    bad = FakeSession([], enter_error=OSError("spawn failed"))

    def factory(spec: McpServerConfig) -> FakeSession:
        return good if spec.command == "good" else bad

    config = McpConfig(servers={
        "ok": McpServerConfig(
            transport="stdio", command="good", tools={"query_ticket_order": _policy()},
        ),
        "bad": McpServerConfig(
            transport="stdio",
            command="bad",
            required=False,
            tools={"query_ticket_order": _policy()},
        ),
    })
    with McpHub(config, client_factory=factory) as hub:
        assert [tool.name for tool in hub.tools] == ["ok__query_ticket_order"]
        assert hub.errors and "bad" in hub.errors[0]
        assert good.exited is False
    assert good.exited


def test_listing_timeout_closes_the_session():
    session = FakeSession([_remote("query_ticket_order")], list_sleep=30)
    config = _config(startup_timeout_seconds=0.3)
    with pytest.raises(McpStartupError, match="没有完成启动"):
        _hub(session, config).connect()
    assert session.entered
    assert session.exited


def test_missing_command_fails_startup():
    config = McpConfig(servers={
        "gone": McpServerConfig(
            transport="stdio",
            command="scenic-mcp-missing-bin",
            startup_timeout_seconds=3,
            tools={"query_ticket_order": _policy()},
        )
    })
    with pytest.raises(McpStartupError, match="gone"):
        McpHub(config).connect()


def test_disabled_config_does_not_open_sessions():
    seen: list[str] = []

    def factory(spec: McpServerConfig) -> FakeSession:
        seen.append(spec.command or "")
        return FakeSession([_remote("query_ticket_order")])

    disabled = McpConfig(enabled=False, servers=_config().servers)
    with McpHub(disabled, client_factory=factory) as hub:
        assert hub.tools == []
    assert seen == []


def test_close_then_call_reports_the_connection_closed():
    session = FakeSession([_remote("query_ticket_order")])
    hub = _hub(session)
    hub.connect()
    runtime = ToolRuntime(hub.tools)
    hub.close()
    result = _execute(runtime, "scenic_ticket__query_ticket_order", {"order_id": "O20261101"})
    assert result.error_code is ToolErrorCode.EXECUTION_ERROR
    assert "关闭" in result.content["error"]
    assert session.exited


def test_invalid_schema_on_a_required_server_fails_startup():
    session = FakeSession([_remote("query_ticket_order", schema={"type": "string"})])
    with pytest.raises(McpStartupError, match="schema"):
        _hub(session).connect()
    assert session.exited


def test_paged_tool_list_is_followed():
    class Paged(FakeSession):
        def __init__(self) -> None:
            super().__init__([])
            self.pages = 0

        async def list_tools(self, cursor: str | None = None, **kwargs: Any) -> SimpleNamespace:
            self.pages += 1
            if cursor is None:
                return SimpleNamespace(tools=[_remote("query_ticket_order")], next_cursor="p2")
            return SimpleNamespace(tools=[_remote("get_opening_hours")], next_cursor=None)

    session = Paged()
    config = _config(tools={
        "query_ticket_order": _policy(),
        "get_opening_hours": _policy(),
    })
    with _hub(session, config) as hub:
        names = sorted(tool.name for tool in hub.tools)
    assert session.pages == 2
    assert names == ["scenic_ticket__get_opening_hours", "scenic_ticket__query_ticket_order"]


def test_agent_loop_executes_mcp_tools_only_through_runtime():
    session = FakeSession(
        [_remote("query_ticket_order")],
        result=SimpleNamespace(
            is_error=False,
            structured_content={"order_id": "O20261101", "status": "已支付"},
            content=[],
        ),
    )
    call = {
        "id": "call_order_1",
        "name": "scenic_ticket__query_ticket_order",
        "arguments": json.dumps({"order_id": "O20261101"}),
    }
    adapter = ScriptedAdapter([
        ModelResult(kind="tool_calls", tool_calls=[call], text=""),
        ModelResult(kind="text", text="订单已支付。", finish_reason="stop"),
    ])
    with _hub(session) as hub:
        runtime = ToolRuntime([*SCENIC_TOOLS, *hub.tools])
        result = run_agent_loop(
            adapter=adapter,
            runtime=runtime,
            ctx=_ctx(),
            system="你是景区客服。",
            user="看下订单 O20261101",
            guard=LoopGuard(max_turns=4),
            on_event=lambda event, payload: None,
        )
    assert result.stop_reason == "protocol_stop"
    assert result.tool_results[0].ok
    assert result.tool_results[0].content["status"] == "已支付"
    assert session.calls[0]["name"] == "query_ticket_order"
    schema_names = [item["function"]["name"] for item in adapter.requests[0].tools or []]
    assert "scenic_ticket__query_ticket_order" in schema_names
    assert "query_ticket_order" not in schema_names
    tool_messages = [item for item in (adapter.requests[1].messages or []) if item.get("role") == "tool"]
    assert tool_messages[0]["tool_call_id"] == "call_order_1"


def test_agent_loop_invalid_mcp_args_never_reach_the_server():
    session = FakeSession([_remote("query_ticket_order")])
    call = {
        "id": "bad",
        "name": "scenic_ticket__query_ticket_order",
        "arguments": json.dumps({"order_id": "nope"}),
    }
    adapter = ScriptedAdapter([
        ModelResult(kind="tool_calls", tool_calls=[call], text=""),
        ModelResult(kind="text", text="订单号不对。", finish_reason="stop"),
    ])
    with _hub(session) as hub:
        result = run_agent_loop(
            adapter=adapter,
            runtime=ToolRuntime(hub.tools),
            ctx=_ctx(),
            system="你是景区客服。",
            user="查单",
            guard=LoopGuard(max_turns=4),
            on_event=lambda event, payload: None,
        )
    assert result.tool_results[0].error_code is ToolErrorCode.INVALID_ARGS
    assert session.calls == []


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_stdio_roundtrip_governance_and_process_exit(tmp_path, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING, logger="scenic_cs_agent.mcp")
    pidfile = tmp_path / "mcp.pid"
    config = McpConfig(servers={
        "scenic_ticket": McpServerConfig(
            transport="stdio",
            command=sys.executable,
            args=["-m", "scenic_cs_agent.mcp_server"],
            env={"SCENIC_MCP_PIDFILE": str(pidfile)},
            startup_timeout_seconds=20,
            tools={
                "query_ticket_order": _policy(timeout_seconds=5),
                "get_opening_hours": _policy(timeout_seconds=5),
            },
        )
    })
    ctx = _ctx()
    with McpHub(config, local_names=[tool.name for tool in SCENIC_TOOLS]) as hub:
        runtime = ToolRuntime([*SCENIC_TOOLS, *hub.tools])
        assert "get_open_hours" in runtime.names(ctx)
        assert "scenic_ticket__get_opening_hours" in runtime.names(ctx)

        order = _execute(runtime, "scenic_ticket__query_ticket_order", {"order_id": "O20261101"}, ctx)
        assert order.ok
        assert order.content["status"] == "已支付"
        assert order.content["scenic_name"] == "云栖山"

        hours = _execute(runtime, "scenic_ticket__get_opening_hours", {"scenic_name": "拙政园"}, ctx)
        assert hours.ok
        assert "07:30" in hours.content["open_hours"]

        local = _execute(runtime, "get_open_hours", {"scenic_name": "拙政园"}, ctx)
        assert local.ok

        missing = _execute(runtime, "scenic_ticket__query_ticket_order", {"order_id": "O00000000"}, ctx)
        assert missing.ok is False
        assert "没有订单" in missing.content["error"]

        invalid = _execute(runtime, "scenic_ticket__query_ticket_order", {"order_id": "nope"}, ctx)
        assert invalid.error_code is ToolErrorCode.INVALID_ARGS

        pid = int(pidfile.read_text(encoding="utf-8"))
        assert _alive(pid)

    deadline = time.time() + 3
    while time.time() < deadline and _alive(pid):
        time.sleep(0.05)
    assert not _alive(pid)
    # 关连接时如果在另一条任务里退出 async with，anyio 会抛 RuntimeError。正常关闭不该打这行。
    assert "关闭 MCP 服务" not in caplog.text


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_port(port: int) -> None:
    deadline = time.time() + 10
    while time.time() < deadline:
        with socket.socket() as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.05)
    raise AssertionError(f"端口 {port} 没有开始监听")


def test_streamable_http_roundtrip():
    port = _free_port()
    thread = threading.Thread(
        target=lambda: mcp_http_server.run(transport="streamable-http", host="127.0.0.1", port=port),
        name="mcp-http-test",
        daemon=True,
    )
    thread.start()
    _wait_port(port)
    config = McpConfig(servers={
        "scenic_ticket": McpServerConfig(
            transport="http",
            url=f"http://127.0.0.1:{port}/mcp",
            headers={"Authorization": None},
            startup_timeout_seconds=10,
            tools={
                "query_ticket_order": _policy(timeout_seconds=5),
                "get_opening_hours": _policy(timeout_seconds=5),
            },
        )
    })
    with McpHub(config) as hub:
        result = _execute(
            ToolRuntime(hub.tools),
            "scenic_ticket__get_opening_hours",
            {"scenic_name": "镜湖湿地"},
        )
    assert result.ok
    assert result.content["scenic_name"] == "镜湖湿地"
