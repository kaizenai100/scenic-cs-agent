"""把 MCP 服务上的工具收成普通 Tool，交给现有的 ToolRuntime。

Agent Loop 只会调用 runtime.execute，这里不另开一条执行通道。
远端工具进 Runtime 之后，参数校验、场景白名单、审批、超时、错误码、审计
和本地工具是同一套。

连接自己占一个事件循环线程：官方 Client 是异步的，Runtime 在线程池里同步调用。
call 把协程提交到这条循环上再等待。超时以 Runtime 的硬超时为准，
这里把协议读超时放得稍长一点，避免先抛异常被收成「执行失败」而不是「超时」。
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import CancelledError as FutureCancelled
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from mcp import Client, StdioServerParameters
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from scenic_cs_agent.config.loader import McpConfig, McpServerConfig, McpToolPolicy
from scenic_cs_agent.tools.base import Tool
from scenic_cs_agent.tools.context import ExecutionContext
from scenic_cs_agent.tools.errors import ToolFailed

logger = logging.getLogger("scenic_cs_agent.mcp")

# 测试可以换成假的异步客户端，只要能当异步上下文管理器用，并且有 list_tools / call_tool。
ClientFactory = Callable[[McpServerConfig], Any]

# 进程退出时还没关的连接放这里，避免 stdio 子进程留下来。
_OPEN_HUBS: set[McpHub] = set()


def _close_open_hubs() -> None:
    for hub in list(_OPEN_HUBS):
        hub.close()


atexit.register(_close_open_hubs)


class McpStartupError(Exception):
    """服务没连上，或允许的工具没法注册。required 的服务出现这个错误时，整个 Hub 失败。"""


class McpServerCrashed(ToolFailed):
    """调用过程中连接断了（进程退出、管道断开、连接已关闭）。文案回给模型。"""


@dataclass(frozen=True)
class McpToolReport:
    """一条远端工具的注册结果。state 是 registered 或 skipped。

    reason 在跳过时才有：not_allowlisted / missing_on_server / name_collision / invalid_schema。
    """

    server: str
    remote_name: str
    exposed_name: str | None
    state: str
    reason: str | None = None


@dataclass(frozen=True)
class _RemoteTool:
    name: str
    description: str
    input_schema: dict[str, Any]


def _present(values: dict[str, str | None]) -> dict[str, str]:
    """丢掉没配的占位（加载后是 None 或空串），避免把空密钥传给子进程或请求头。"""
    return {key: value for key, value in values.items() if value}


def stdio_parameters(spec: McpServerConfig) -> StdioServerParameters:
    """stdio 启动参数。环境变量只附加配置里有值的那些，其余沿用 SDK 的默认继承。"""
    if not spec.command:
        raise McpStartupError("stdio 传输缺少 command")
    env = _present(spec.env)
    return StdioServerParameters(
        command=spec.command,
        args=list(spec.args),
        env=env or None,
        cwd=spec.cwd,
    )


@asynccontextmanager
async def _stdio_session(spec: McpServerConfig):
    async with Client(stdio_parameters(spec), read_timeout_seconds=spec.startup_timeout_seconds) as client:
        yield client


@asynccontextmanager
async def _http_session(spec: McpServerConfig):
    # 用到 HTTP 才导入。只跑 stdio 的测试不必碰 httpx2。
    import httpx2
    from mcp.client.streamable_http import streamable_http_client

    if not spec.url:
        raise McpStartupError("http 传输缺少 url")
    headers = _present(spec.headers)
    # 读超时留长：服务端可能把响应流挂着。真正的工具超时在 ToolRuntime。
    timeout = httpx2.Timeout(30.0, read=300.0)
    async with httpx2.AsyncClient(headers=headers, timeout=timeout) as http:
        transport = streamable_http_client(spec.url, http_client=http)
        async with Client(transport, read_timeout_seconds=spec.startup_timeout_seconds) as client:
            yield client


def _open_session(spec: McpServerConfig):
    if spec.transport == "http":
        return _http_session(spec)
    return _stdio_session(spec)


def _disconnected(exc: BaseException) -> bool:
    """管道断了、进程没了。超时不是断开，超时交给 ToolRuntime 记成 TIMEOUT。"""
    if isinstance(exc, BaseExceptionGroup):
        return any(_disconnected(item) for item in exc.exceptions)
    if isinstance(exc, (BrokenPipeError, ConnectionError, EOFError)):
        return True
    return type(exc).__name__ in {"BrokenResourceError", "ClosedResourceError", "EndOfStream"}


def _content_text(content: Any) -> str:
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, dict):
            text = block.get("text") if block.get("type", "text") == "text" else None
        else:
            text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)
    return "\n".join(parts)


def _clip(text: str) -> str:
    text = text.strip()
    if len(text) > 500:
        return text[:500] + "…"
    return text


def _payload(result: Any, remote_name: str) -> dict[str, Any]:
    """把 MCP 的 CallToolResult 收成给模型的 dict。is_error 变成 ToolFailed，不当成成功。"""
    if isinstance(result, dict):
        is_error = bool(result.get("is_error") or result.get("isError"))
        structured = result.get("structured_content", result.get("structuredContent"))
        content = result.get("content") or []
    else:
        is_error = bool(getattr(result, "is_error", False) or getattr(result, "isError", False))
        structured = getattr(result, "structured_content", None)
        if structured is None:
            structured = getattr(result, "structuredContent", None)
        content = getattr(result, "content", None) or []

    if is_error:
        text = _content_text(content)
        prefix = f"Error executing tool {remote_name}: "
        if text.startswith(prefix):
            text = text[len(prefix):]
        raise ToolFailed(_clip(text) or f"MCP 工具 {remote_name} 返回错误")

    if isinstance(structured, dict):
        return structured
    text = _content_text(content)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"text": text}
    if isinstance(parsed, dict):
        return parsed
    return {"result": parsed}


def _parse_remote(raw: Any) -> _RemoteTool:
    if isinstance(raw, dict):
        name = raw.get("name")
        description = raw.get("description") or ""
        schema = raw.get("input_schema", raw.get("inputSchema"))
    else:
        name = getattr(raw, "name", None)
        description = getattr(raw, "description", None) or ""
        schema = getattr(raw, "input_schema", None)
        if schema is None:
            schema = getattr(raw, "inputSchema", None)
    if not isinstance(name, str) or not name:
        raise ValueError("工具缺少 name")
    if hasattr(schema, "model_dump"):
        schema = schema.model_dump(by_alias=True, exclude_none=True)
    if not isinstance(schema, dict):
        raise ValueError(f"工具 {name} 没有 inputSchema")
    return _RemoteTool(name=name, description=str(description), input_schema=schema)


def _args_model(schema: dict[str, Any]) -> type[BaseModel]:
    """用远端的 JSON Schema 做执行前校验，校验点仍是 ToolRuntime 里的 model_validate。

    不自己放行参数。schema 不合法或不是 object，就拒绝注册这把工具。
    """
    if schema.get("type") != "object":
        raise ValueError("inputSchema.type 必须是 object")
    Draft202012Validator.check_schema(schema)
    checker = Draft202012Validator(schema)

    class Args(BaseModel):
        model_config = ConfigDict(extra="allow")

        @model_validator(mode="before")
        @classmethod
        def _check(cls, data: Any) -> Any:
            if not isinstance(data, dict):
                raise ValueError("参数必须是 JSON 对象")
            errors = sorted(checker.iter_errors(data), key=lambda err: [str(part) for part in err.path])
            if not errors:
                return data
            first = errors[0]
            where = ".".join(str(part) for part in first.path) or "参数"
            raise ValueError(f"{where}：{first.message}")

    Args.__name__ = "McpArgs"
    return Args


def _needs_approval(policy: McpToolPolicy) -> bool:
    if policy.requires_approval is not None:
        return policy.requires_approval
    # 没写明时，写操作先卡住等审批。只读查询可以直接跑。
    return not policy.read_only


class McpConnection:
    """一条 MCP 会话。connect / call_tool / close 是同步的，里面用单独的事件循环。

    官方 Client 用了 anyio 的 cancel scope，必须在进入它的那条任务里退出。
    所以会话任务会一直待在 async with 里，直到 close 通知它离开；
    不能在另一条任务上 aclose。
    """

    def __init__(self, name: str, spec: McpServerConfig, factory: ClientFactory) -> None:
        self.name = name
        self.spec = spec
        self._factory = factory
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._loop_ready = threading.Event()
        self._listed = threading.Event()
        self._client: Any = None
        self._stop: asyncio.Event | None = None
        self._session_fut: Future[None] | None = None
        self._closed = False
        self._lock = threading.Lock()
        self._inflight: set[Future[Any]] = set()

    def connect(self) -> list[Any]:
        """拉起会话并列出工具。失败时会把已经启动的进程关掉，再抛 McpStartupError。"""
        if self._thread is not None:
            raise RuntimeError(f"MCP 服务 {self.name} 已经连接过")
        started: Future[list[Any]] = Future()
        try:
            self._start_loop()
            # 会话任务不放进 _inflight。close 会取消进行中的调用，但不能把会话本身一起取消掉。
            self._session_fut = self._submit(self._session(started), track=False)
            return started.result(timeout=self.spec.startup_timeout_seconds)
        except McpStartupError:
            self.close()
            raise
        except FutureTimeout:
            self.close()
            raise McpStartupError(
                f"MCP 服务 {self.name} 在 {self.spec.startup_timeout_seconds:g}s 内没有完成启动或列出工具"
            ) from None
        except Exception as exc:
            self.close()
            raise McpStartupError(f"MCP 服务 {self.name} 启动失败：{type(exc).__name__}: {exc}") from exc

    def call_tool(self, remote_name: str, arguments: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
        if self._closed or self._client is None or self._loop is None or not self._loop.is_running():
            raise McpServerCrashed(f"MCP 服务 {self.name} 的连接已关闭")
        # 比 ToolRuntime 的硬超时多等一会儿。挂起的调用由 Runtime 记成超时或结果未知，
        # 这里如果先超时，异常类型会把「超时」误收成「执行失败」。
        try:
            return self._run(
                self._invoke(remote_name, arguments, timeout_s),
                timeout=timeout_s + 1.5,
            )
        except (ToolFailed, McpServerCrashed):
            raise
        except FutureTimeout:
            raise McpServerCrashed(f"MCP 服务 {self.name} 调用 {remote_name} 没有在时限内返回")
        except FutureCancelled:
            raise McpServerCrashed(f"MCP 服务 {self.name} 的连接已关闭")
        except Exception as exc:
            if _disconnected(exc):
                raise McpServerCrashed(f"MCP 服务 {self.name} 在调用 {remote_name} 时断开") from exc
            raise

    def close(self) -> None:
        """关掉会话。stdio 会停掉子进程。可以重复调用。"""
        if self._closed:
            return
        self._closed = True
        loop = self._loop
        thread = self._thread
        with self._lock:
            inflight = list(self._inflight)
        for fut in inflight:
            fut.cancel()
        if loop is None:
            return
        session = self._session_fut
        if loop.is_running() and session is not None:
            # 工具还没列出来时，会话多半卡在握手或 list_tools，设事件到不了 await。
            # 取消的是持有 async with 的那条任务，退出也发生在它里面。
            if not self._listed.is_set():
                session.cancel()
            elif self._stop is not None:
                loop.call_soon_threadsafe(self._stop.set)
            try:
                session.result(timeout=8)
            except FutureTimeout:
                session.cancel()
                try:
                    session.result(timeout=2)
                except Exception:
                    logger.warning("关闭 MCP 服务 %s 超时", self.name)
            except FutureCancelled:
                pass
            except Exception as exc:
                # 还没列出工具就失败时，异常已经交给 connect 了，这里再打一条是重复的。
                if self._listed.is_set():
                    logger.warning("关闭 MCP 服务 %s 出错：%s", self.name, type(exc).__name__)
            loop.call_soon_threadsafe(loop.stop)
        elif not loop.is_closed():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)
            if thread.is_alive():
                logger.warning("MCP 服务 %s 的事件循环线程没有退出", self.name)

    def _start_loop(self) -> None:
        self._loop = asyncio.new_event_loop()

        def _serve() -> None:
            assert self._loop is not None
            loop = self._loop
            asyncio.set_event_loop(loop)
            # 循环真正开始处理回调后再放行，避免 close 时它还没跑起来。
            loop.call_soon(self._loop_ready.set)
            try:
                loop.run_forever()
            finally:
                pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.close()

        self._thread = threading.Thread(target=_serve, name=f"mcp-{self.name}", daemon=True)
        self._thread.start()
        if not self._loop_ready.wait(timeout=2):
            raise McpStartupError(f"MCP 服务 {self.name} 的事件循环没有启动")

    def _submit(self, coro: Any, *, track: bool = True) -> Future[Any]:
        if self._loop is None:
            raise McpServerCrashed(f"MCP 服务 {self.name} 的连接已关闭")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        if track:
            with self._lock:
                self._inflight.add(fut)
        return fut

    def _run(self, coro: Any, *, timeout: float) -> Any:
        fut = self._submit(coro)
        try:
            return fut.result(timeout=timeout)
        except FutureTimeout:
            fut.cancel()
            try:
                fut.result(timeout=5)
            except Exception:
                pass
            raise
        finally:
            with self._lock:
                self._inflight.discard(fut)

    async def _session(self, started: Future[list[Any]]) -> None:
        """持有客户端的那条任务。离开 async with 也在这里，不能换一条任务去关。"""
        self._stop = asyncio.Event()
        try:
            async with self._factory(self.spec) as client:
                self._client = client
                try:
                    tools = await _list_tools(client)
                except Exception as exc:
                    if not started.done():
                        started.set_exception(exc)
                    return
                self._listed.set()
                if not started.done():
                    started.set_result(tools)
                await self._stop.wait()
        except asyncio.CancelledError:
            if not started.done():
                started.cancel()
            raise
        except Exception as exc:
            if not started.done():
                started.set_exception(exc)
            raise
        finally:
            self._client = None

    async def _invoke(self, remote_name: str, arguments: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        client = self._client
        if client is None or self._closed:
            raise McpServerCrashed(f"MCP 服务 {self.name} 的连接已关闭")
        try:
            result = await client.call_tool(
                remote_name,
                arguments,
                read_timeout_seconds=timeout_s + 1.0,
            )
        except Exception as exc:
            if _disconnected(exc):
                raise McpServerCrashed(f"MCP 服务 {self.name} 在调用 {remote_name} 时断开") from exc
            raise
        return _payload(result, remote_name)


async def _list_tools(client: Any) -> list[Any]:
    collected: list[Any] = []
    cursor: str | None = None
    for _ in range(20):
        page = await client.list_tools(cursor=cursor) if cursor else await client.list_tools()
        if isinstance(page, dict):
            batch = page.get("tools")
            cursor = page.get("next_cursor")
        else:
            batch = getattr(page, "tools", None)
            cursor = getattr(page, "next_cursor", None)
        if batch is None:
            raise McpStartupError("list_tools 的返回里没有 tools")
        collected.extend(batch)
        if not cursor:
            return collected
    raise McpStartupError("list_tools 分页超过 20 页，已停止")


def _build_tool(
    server: str,
    remote: _RemoteTool,
    policy: McpToolPolicy,
    exposed: str,
    connection: McpConnection,
) -> Tool:
    description = (policy.description or remote.description or remote.name).strip()
    description = f"[{server}] {description}"
    timeout_s = policy.timeout_seconds
    remote_name = remote.name
    args_model = _args_model(remote.input_schema)

    def func(args: BaseModel, ctx: ExecutionContext) -> dict[str, Any]:
        # 用户身份留在 ctx，不放进工具参数。run_id 只进调试日志，方便和审计对上。
        logger.debug("mcp call server=%s tool=%s run_id=%s", server, remote_name, ctx.run_id)
        return connection.call_tool(remote_name, args.model_dump(mode="json"), timeout_s=timeout_s)

    return Tool(
        name=exposed,
        description=description,
        args_model=args_model,
        func=func,
        read_only=policy.read_only,
        timeout_s=timeout_s,
        requires_approval=_needs_approval(policy),
        scenes=frozenset(policy.scenes),
        sensitive_fields=frozenset(policy.sensitive_fields),
    )


def _register(
    server: str,
    spec: McpServerConfig,
    raw_tools: list[Any],
    taken: set[str],
    connection: McpConnection,
    separator: str,
) -> tuple[list[Tool], list[McpToolReport]]:
    remotes: list[_RemoteTool] = []
    seen: set[str] = set()
    reports: list[McpToolReport] = []
    for raw in raw_tools:
        try:
            remote = _parse_remote(raw)
        except (ValueError, SchemaError) as exc:
            if spec.required:
                raise McpStartupError(f"MCP 服务 {server} 的工具定义无法使用：{exc}") from exc
            reports.append(McpToolReport(server, "?", None, "skipped", "invalid_schema"))
            continue
        if remote.name in seen:
            if spec.required:
                raise McpStartupError(f"MCP 服务 {server} 返回了重复工具名 {remote.name}")
            continue
        seen.add(remote.name)
        remotes.append(remote)

    by_name = {remote.name: remote for remote in remotes}
    missing = [name for name in spec.tools if name not in by_name]
    if missing and spec.required:
        raise McpStartupError(f"MCP 服务 {server} 没有列出允许的工具：{', '.join(missing)}")
    for name in missing:
        reports.append(McpToolReport(server, name, None, "skipped", "missing_on_server"))

    tools: list[Tool] = []
    for remote in remotes:
        policy = spec.tools.get(remote.name)
        if policy is None:
            reports.append(McpToolReport(server, remote.name, None, "skipped", "not_allowlisted"))
            continue
        exposed = f"{server}{separator}{remote.name}"
        if exposed in taken:
            if spec.required:
                raise McpStartupError(
                    f"MCP 工具 {exposed} 与已有工具重名。"
                    "远端工具会加上服务名前缀，仍然冲突时拒绝注册，避免盖住已有工具。"
                )
            reports.append(McpToolReport(server, remote.name, exposed, "skipped", "name_collision"))
            continue
        try:
            tool = _build_tool(server, remote, policy, exposed, connection)
        except (ValueError, SchemaError, ValidationError) as exc:
            if spec.required:
                raise McpStartupError(
                    f"MCP 服务 {server} 的工具 {remote.name} 参数 schema 无法校验：{exc}"
                ) from exc
            reports.append(McpToolReport(server, remote.name, exposed, "skipped", "invalid_schema"))
            continue
        taken.add(exposed)
        tools.append(tool)
        reports.append(McpToolReport(server, remote.name, exposed, "registered", None))
    return tools, reports


class McpHub:
    """按配置连接一组 MCP 服务，得到可以直接放进 ToolRuntime 的工具列表。"""

    def __init__(
        self,
        config: McpConfig,
        *,
        local_names: Iterable[str] = (),
        client_factory: ClientFactory | None = None,
    ) -> None:
        self.config = config
        self._local_names = set(local_names)
        self._factory = client_factory or _open_session
        self._connections: list[McpConnection] = []
        self.tools: list[Tool] = []
        self.reports: list[McpToolReport] = []
        # 服务名 → connected / disabled / error。就绪检查读这份，不自己再去连。
        self.server_state: dict[str, str] = {}
        # 只在 error 时有值。说明给运维看，不要把密钥写进来。
        self.server_errors: dict[str, str] = {}
        # 非 required 的服务没连上时记在这里，不抛。
        self.errors: list[str] = []
        self._did_connect = False
        self._closed = False

    def connect(self) -> list[Tool]:
        """连接 enabled 的服务。required 失败会关掉已经连上的，再抛 McpStartupError。"""
        if self._did_connect:
            raise RuntimeError("McpHub.connect 只能调用一次")
        self._did_connect = True
        _OPEN_HUBS.add(self)
        if not self.config.enabled:
            for name in self.config.servers:
                self.server_state[name] = "disabled"
            return []
        taken = set(self._local_names)
        try:
            for name, spec in self.config.servers.items():
                if not spec.enabled:
                    self.server_state[name] = "disabled"
                    continue
                self._connect_server(name, spec, taken)
        except Exception:
            # 失败时不把半截工具交出去：连接会关，调用它们只会得到「连接已关闭」。
            self.tools.clear()
            self.close()
            raise
        logger.info(
            "MCP 已注册 %d 个工具：%s",
            len(self.tools),
            ", ".join(tool.name for tool in self.tools) or "（无）",
        )
        return list(self.tools)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _OPEN_HUBS.discard(self)
        connections = list(self._connections)
        self._connections.clear()
        for connection in connections:
            connection.close()

    def __enter__(self) -> McpHub:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _connect_server(self, name: str, spec: McpServerConfig, taken: set[str]) -> None:
        connection = McpConnection(name, spec, self._factory)
        try:
            raw_tools = connection.connect()
        except McpStartupError as exc:
            if spec.required:
                raise
            message = str(exc)
            self.errors.append(message)
            self.server_state[name] = "error"
            self.server_errors[name] = message
            logger.warning("%s", exc)
            return
        self._connections.append(connection)
        tools, reports = _register(
            name, spec, raw_tools, taken, connection, self.config.name_separator
        )
        self.tools.extend(tools)
        self.reports.extend(reports)
        self.server_state[name] = "connected"


__all__ = [
    "McpHub",
    "McpServerCrashed",
    "McpStartupError",
    "McpToolReport",
    "stdio_parameters",
]
