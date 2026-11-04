"""把一次用户消息跑成一轮 Agent，并把过程写成 RunEvent。

模型只经过 Gateway。工具只经过 ToolRuntime，能看见哪些由 scene 决定。
这里不自己读密钥，也不直接调用工具函数。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import threading
from typing import Any

from scenic_cs_agent.api.events import GENERIC_FAILURE, tool_completed_data, tool_started_data
from scenic_cs_agent.api.run_store import RunState, RunStore
from scenic_cs_agent.gateway.errors import GatewayError
from scenic_cs_agent.prompts.registry import PromptRegistry
from scenic_cs_agent.prompts.scenic_presale import render_scenic_presale
from scenic_cs_agent.runtime.agent_loop import AgentLoopResult, run_agent_loop
from scenic_cs_agent.runtime.loop_guard import LoopGuard
from scenic_cs_agent.runtime.model_types import ModelAdapter
from scenic_cs_agent.tools.context import ExecutionContext
from scenic_cs_agent.tools.runtime import ToolRuntime

logger = logging.getLogger("scenic_cs_agent.agent")

# 和流式生产者用同一个原因，取消事件的 data.reason 保持稳定。
CANCEL_REASON = "client_cancel"

# 失败码只允许这种短 token，避免把异常文本送进 SSE。
_CODE = re.compile(r"^[a-z0-9_.-]{1,40}$")

# 正文按这个长度切开，订阅方能看到增量，而不是等整段一起到。
_TEXT_CHUNK = 8

# 身份由服务端填。请求正文里没有这些字段，模型也填不进来。
_ANONYMOUS_USER = "anonymous"
_CHANNEL = "api"

# 护栏停下时给客户端的固定说明。不要拼模型原文。
_GUARD = {
    "max_turns": "对话轮数已达上限",
    "duplicate_action": "重复调用了相同的工具",
}

_HINTS = {
    "missing_api_key": "模型没有配置密钥",
    "rate_limited": "请求过于频繁，请稍后重试",
    "circuit_open": "模型暂时不可用，请稍后重试",
    "timeout": "上游暂时不可用，请稍后重试",
    "connection": "上游暂时不可用，请稍后重试",
    "429": "上游暂时不可用，请稍后重试",
    "500": "上游暂时不可用，请稍后重试",
    "502": "上游暂时不可用，请稍后重试",
    "503": "上游暂时不可用，请稍后重试",
    "504": "上游暂时不可用，请稍后重试",
    **_GUARD,
}


class AgentService:
    """进程里的对话入口。Gateway、工具和 prompt 都在启动时放进来，请求路径上不再装配。"""

    def __init__(
        self,
        *,
        gateway: ModelAdapter,
        runtime: ToolRuntime,
        prompts: PromptRegistry,
        store: RunStore,
    ) -> None:
        # 生产环境放进来的是进程内 Gateway，它实现了 ModelAdapter.generate。
        self._gateway = gateway
        self._runtime = runtime
        self._prompts = prompts
        self._store = store

    async def run(
        self,
        state: RunState,
        *,
        message: str,
        system: str | None,
        scene: str,
    ) -> None:
        """跑完这一次 Run。取消会先写成 run.cancelled，再把 CancelledError 抛出去。"""

        started = await self._store.append(state, "run.started", {})
        if started is None:
            await self._store.mark_cancelled(state, reason=CANCEL_REASON)
            return

        ctx = ExecutionContext(
            run_id=state.run_id,
            user_id=_ANONYMOUS_USER,
            channel=_CHANNEL,
            scene=scene,
        )
        try:
            system_text = system if system else self._system_text(ctx)
        except Exception as exc:
            logger.warning("run %s prompt 渲染失败 exc=%s", state.run_id, type(exc).__name__)
            await self._fail(state, dict(GENERIC_FAILURE))
            return

        events: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()
        thread_cancel = threading.Event()
        loop = asyncio.get_running_loop()
        drainer = asyncio.create_task(self._drain(events, state), name=f"drain-{state.run_id}")
        watcher = asyncio.create_task(
            self._watch_cancel(state, thread_cancel),
            name=f"cancel-{state.run_id}",
        )
        try:
            try:
                result = await asyncio.to_thread(
                    self._execute_loop,
                    ctx,
                    system_text,
                    message,
                    thread_cancel,
                    loop,
                    events,
                )
                await self._stop_drain(drainer, events)
                await self._finish(state, result)
            except asyncio.CancelledError:
                thread_cancel.set()
                _clear_cancellation()
                with contextlib.suppress(Exception):
                    await self._stop_drain(drainer, events)
                await self._mark_cancelled(state)
                raise
            except Exception as exc:
                logger.warning("run %s 失败 exc=%s", state.run_id, type(exc).__name__)
                with contextlib.suppress(Exception):
                    await self._stop_drain(drainer, events)
                await self._fail(state, _public_failure(exc))
        finally:
            thread_cancel.set()
            if not watcher.done():
                watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher

    def _system_text(self, ctx: ExecutionContext) -> str:
        rendered = render_scenic_presale(
            allowed_tools=self._runtime.names(ctx),
            registry=self._prompts,
        )
        return rendered.text

    def _execute_loop(
        self,
        ctx: ExecutionContext,
        system: str,
        user: str,
        thread_cancel: threading.Event,
        loop: asyncio.AbstractEventLoop,
        events: asyncio.Queue[tuple[str, dict[str, Any]] | None],
    ) -> AgentLoopResult:
        """在工作线程里跑同步的 Agent Loop。工具事件丢回事件循环，线程自己不写 Store。"""

        def post(item: tuple[str, dict[str, Any]] | None) -> None:
            try:
                loop.call_soon_threadsafe(events.put_nowait, item)
            except RuntimeError:
                # 事件循环已经关了。没有人再读这条。
                return

        runtime = _ToolEvents(self._runtime, post)
        try:
            return run_agent_loop(
                adapter=self._gateway,
                runtime=runtime,  # type: ignore[arg-type]
                ctx=ctx,
                system=system,
                user=user,
                guard=LoopGuard(max_turns=8),
                cancel_event=thread_cancel,
                on_event=_log_loop_event,
            )
        finally:
            post(None)

    async def _drain(
        self,
        events: asyncio.Queue[tuple[str, dict[str, Any]] | None],
        state: RunState,
    ) -> None:
        while True:
            item = await events.get()
            if item is None:
                return
            kind, payload = item
            if kind == "tool.started":
                await self._store.append(
                    state,
                    "tool.started",
                    tool_started_data(str(payload["id"]), str(payload["name"])),
                )
            elif kind == "tool.completed":
                await self._store.append(
                    state,
                    "tool.completed",
                    tool_completed_data(
                        str(payload["id"]),
                        str(payload["name"]),
                        ok=bool(payload["ok"]),
                    ),
                )

    async def _stop_drain(
        self,
        drainer: asyncio.Task[None],
        events: asyncio.Queue[tuple[str, dict[str, Any]] | None],
    ) -> None:
        if not drainer.done():
            events.put_nowait(None)
            await drainer
            return
        if drainer.cancelled():
            return
        error = drainer.exception()
        if error is not None:
            raise error

    async def _finish(self, state: RunState, result: AgentLoopResult) -> None:
        logger.info(
            "run %s 结束 stop_reason=%s turns=%s",
            state.run_id,
            result.stop_reason,
            result.turns,
        )
        if result.stop_reason == "cancelled" or state.cancel_event.is_set():
            await self._mark_cancelled(state)
            return
        if result.stop_reason != "protocol_stop":
            hint = _GUARD.get(result.stop_reason, "生成未能正常结束")
            code = result.stop_reason if result.stop_reason in _GUARD else "internal"
            await self._fail(
                state,
                {"code": code, "stage": "guard", "retryable": False, "hint": hint},
            )
            return
        text = result.text or ""
        for chunk in _chunks(text):
            if state.cancel_event.is_set():
                await self._mark_cancelled(state)
                return
            wrote = await self._store.append(state, "text.delta", {"delta": chunk})
            if wrote is None:
                await self._mark_cancelled(state)
                return
        done = await self._store.complete(state, finish_reason="stop", usage=None)
        if not done:
            await self._mark_cancelled(state)

    async def _fail(self, state: RunState, failure: dict[str, Any]) -> None:
        wrote = await self._store.fail(state, failure)
        if not wrote:
            await self._mark_cancelled(state)

    async def _mark_cancelled(self, state: RunState) -> None:
        try:
            await self._store.mark_cancelled(state, reason=CANCEL_REASON)
        except Exception as exc:
            logger.warning("写 run.cancelled 失败 exc=%s", type(exc).__name__)

    @staticmethod
    async def _watch_cancel(state: RunState, thread_cancel: threading.Event) -> None:
        await state.cancel_event.wait()
        thread_cancel.set()


class _ToolEvents:
    """包一层 Runtime：执行前后把工具事件送进队列。真正的校验和执行仍在 ToolRuntime。"""

    def __init__(
        self,
        inner: ToolRuntime,
        post: Any,
    ) -> None:
        self._inner = inner
        self._post = post

    def schemas(self, ctx: ExecutionContext) -> list[dict[str, Any]]:
        return self._inner.schemas(ctx)

    def execute(self, call: dict[str, Any], ctx: ExecutionContext, *, approved: bool = False) -> Any:
        self._post(("tool.started", {"id": call["id"], "name": call["name"]}))
        result = self._inner.execute(call, ctx, approved=approved)
        self._post(
            (
                "tool.completed",
                {"id": result.tool_call_id, "name": result.name, "ok": result.ok},
            )
        )
        return result


def _chunks(text: str) -> list[str]:
    if not text:
        return []
    return [text[index : index + _TEXT_CHUNK] for index in range(0, len(text), _TEXT_CHUNK)]


def _public_failure(exc: BaseException) -> dict[str, Any]:
    """异常 → 可以给客户端的四个字段。不读异常消息。"""

    if isinstance(exc, GatewayError) and _CODE.fullmatch(exc.code):
        return {
            "code": exc.code,
            "stage": "generate",
            "retryable": bool(exc.retryable),
            "hint": _HINTS.get(exc.code, "生成失败，请稍后重试"),
        }
    return dict(GENERIC_FAILURE)


def _log_loop_event(event: str, payload: dict[str, Any]) -> None:
    logger.debug("loop %s %s", event, payload.get("stop_reason") or payload.get("tool") or "")


def _clear_cancellation() -> None:
    """清掉当前任务上的取消计数，好让清理阶段的 await 真的能跑完。

    3.12 里任务被 cancel 之后，捕获了 CancelledError 还不够：
    下一个 await 会立刻再次抛出，run.cancelled 就写不进 Store。
    清理写完后，调用方仍然要重新抛出 CancelledError。
    """

    task = asyncio.current_task()
    if task is None:
        return
    while task.uncancel():
        pass
