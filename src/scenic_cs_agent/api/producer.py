"""把一次模型流收成 RunEvent。

生产对话不走这里，走 AgentService。这条路径留给测试：
注入一个脚本模型，覆盖断连、续传、取消和失败收口。

生产者和 HTTP 订阅者是两条任务：
  POST /v1/runs 把本协程丢到后台，马上返回 run_id；
  GET  /v1/runs/{id}/events 只读 RunStore，不直接碰模型。

这样做的原因：用户关掉页面只结束订阅，生成还要继续，
稍后还能用 Last-Event-ID 接上。取消必须走单独的接口。

失败策略分界在「第一条已经写进 Store 的 text.delta」：
  - 之前：可重试的错误可以再来一次（并先写 run.retrying）；
  - 之后：只能再写一条 run.failed。此时如果把异常抛出任务，
    订阅那一头已经把 200 和正文推给客户端了，对方只会看见连接突然断掉。
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Iterator
from typing import Any, Protocol

from scenic_cs_agent.api.events import GENERIC_FAILURE
from scenic_cs_agent.api.run_store import RunState, RunStore
from scenic_cs_agent.runtime.model_types import ModelRequest, ModelStreamEvent

logger = logging.getLogger(__name__)

# 第一条正文之前最多跑这么多次。2 = 首次 + 一次重试。
DEFAULT_MAX_PRE_TEXT_ATTEMPTS = 2

# 用户点了取消接口。写进 run.cancelled.reason，不把 asyncio 的异常文本传出去。
CANCEL_REASON_CLIENT = "client_cancel"

# 模型正常说完。tool_calls 不在这里：本进程没有工具循环，不能当成成功。
_SUCCESS_FINISH_REASONS = frozenset({"stop", None})


class ModelStreamSource(Protocol):
    """任何能按 ModelRequest 吐出 ModelStreamEvent 的东西。

    真实适配器是同步生成器（官方 SDK 不是 async）。
    测试里用脚本模型实现同一个方法，这样生产者不用分两套代码。
    """

    def stream(self, request: ModelRequest) -> Iterator[ModelStreamEvent]:
        """同步迭代。禁止在事件循环线程里直接跑完它。"""


class StreamFailure(Exception):
    """生产者认识的失败。字段都会进 run.failed，所以不许带密钥、堆栈或提示词。

    retryable 描述错误种类（限流、上游 5xx、连不上），不是「这一次 Run 还会自动再试」。
    正文已经发出去之后，即使 retryable 为真，这次 Run 也只补 run.failed。
    """

    def __init__(self, code: str, *, stage: str, retryable: bool, hint: str) -> None:
        super().__init__(code)
        self.code = code
        self.stage = stage
        self.retryable = retryable
        self.hint = hint

    def as_public(self) -> dict[str, Any]:
        """四个公开字段。不要往这里加 str(self) 或 __cause__。"""

        return {
            "code": self.code,
            "stage": self.stage,
            "retryable": self.retryable,
            "hint": self.hint,
        }


def failure_payload(exc: BaseException) -> dict[str, Any]:
    """异常 → 公开失败。只看类型和 StreamFailure 上的显式字段，不读异常消息。

    上游 SDK 的 str(exc) 经常包含请求 URL。公开仓库的响应里不能出现这些。
    类型名对应：限流 / 超时 / 连接失败 / 5xx 可以换一次；
    认证、参数、拒绝类错误再试也是同一个结果，retryable 为假。
    """

    if isinstance(exc, StreamFailure):
        return exc.as_public()
    name = type(exc).__name__
    if name in {"RateLimitError", "APITimeoutError", "APIConnectionError"}:
        return {
            "code": "upstream",
            "stage": "connect",
            "retryable": True,
            "hint": "上游暂时不可用，请稍后重试",
        }
    if name == "InternalServerError":
        return {
            "code": "upstream",
            "stage": "generate",
            "retryable": True,
            "hint": "上游暂时不可用，请稍后重试",
        }
    if name in {
        "AuthenticationError",
        "PermissionDeniedError",
        "BadRequestError",
        "UnprocessableEntityError",
    }:
        return {
            "code": "rejected",
            "stage": "connect",
            "retryable": False,
            "hint": "请求被拒绝",
        }
    return dict(GENERIC_FAILURE)


def map_finish_failure(finish_reason: str | None) -> dict[str, Any] | None:
    """把模型的结束原因翻译成业务失败。返回 None 表示可以按成功收尾。

    流结束不等于成功：length 是被截断，content_filter 是被拒绝。
    这两种都不在重试白名单里（不是 429 / 5xx / 连不上）。
    tool_calls 当前没有执行器，如果当成 stop，客户端会收到一条空白的成功消息。
    """

    if finish_reason in _SUCCESS_FINISH_REASONS:
        return None
    if finish_reason == "length":
        return {
            "code": "truncated",
            "stage": "generate",
            "retryable": False,
            "hint": "回复被长度限制截断",
        }
    if finish_reason == "content_filter":
        return {
            "code": "refusal",
            "stage": "generate",
            "retryable": False,
            "hint": "该内容无法继续生成",
        }
    if finish_reason == "tool_calls":
        return {
            "code": "unsupported",
            "stage": "generate",
            "retryable": False,
            "hint": "当前还不能执行工具调用",
        }
    return {
        "code": "upstream",
        "stage": "generate",
        "retryable": False,
        "hint": "生成未能正常结束",
    }


def _clear_cancellation() -> None:
    """清掉当前任务上的取消计数，好让清理阶段的 await 真的能跑完。

    3.12 里任务被 cancel 之后，捕获了 CancelledError 还不够：
    下一个 await 会立刻再次抛出，run.cancelled 就写不进 Store。
    清理写完后，调用方仍然要重新抛出 CancelledError，任务不能伪装成正常结束。
    """

    task = asyncio.current_task()
    if task is None:
        return
    while task.uncancel():
        pass


async def iterate_model_stream(
    source: ModelStreamSource,
    request: ModelRequest,
    cancel_event: asyncio.Event,
) -> Any:
    """在线程里跑同步 stream()，每拿到一条就放进队列。

    不能用一次 asyncio.to_thread 把整个生成器包起来：那样要等模型全部说完，
    队列里才会出现第一条，客户端看不到逐字输出。
    取消时不在这里置 cancel_event。那个开关只由取消接口打开，
    否则正常结束的 finally 会把成功的 Run 误标成取消。
    """

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    # 生成器退出后线程还可能再投递。这个旗标让迟到的投递直接丢掉。
    stop = threading.Event()

    def post(item: tuple[str, Any]) -> None:
        if stop.is_set():
            return
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:
            # 事件循环已经关了（测试进程正在退出）。没有人再读这条。
            return

    def worker() -> None:
        iterator: Iterator[ModelStreamEvent] | None = None
        try:
            iterator = source.stream(request)
            for event in iterator:
                # 协作式取消：模型还在吐的时候就能停，不必干等 task.cancel。
                if stop.is_set() or cancel_event.is_set():
                    post(("cancelled", None))
                    return
                post(("event", event))
            post(("done", None))
        except Exception as exc:
            # 只投递异常对象。消息文本留在对象里，事件循环那一侧不会把它写进 RunEvent。
            post(("error", exc))
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug("关闭模型流失败")

    thread = threading.Thread(target=worker, name="model-stream", daemon=True)
    thread.start()
    try:
        while True:
            kind, payload = await queue.get()
            if kind == "event":
                if cancel_event.is_set():
                    return
                yield payload
            elif kind in {"done", "cancelled"}:
                return
            else:
                raise payload
    finally:
        stop.set()


async def _consume_once(
    store: RunStore,
    state: RunState,
    source: ModelStreamSource,
    request: ModelRequest,
) -> None:
    """消费一次模型流。成功、业务失败或取消都会在这里写成终态事件。

    传输层异常不在这里吞，交给外层决定能不能在首个 text.delta 之前重试。
    """

    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    fail_info: dict[str, Any] | None = None

    async for event in iterate_model_stream(source, request, state.cancel_event):
        if state.cancel_event.is_set():
            await store.mark_cancelled(state, reason=CANCEL_REASON_CLIENT)
            return
        if event.type == "text.delta":
            delta = event.data.get("delta") or ""
            if not isinstance(delta, str) or not delta:
                continue
            # 重新包一层，只留 delta。模型事件上其余字段不进入 RunEvent。
            wrote = await store.append(state, "text.delta", {"delta": delta})
            if wrote is None:
                await store.mark_cancelled(state, reason=CANCEL_REASON_CLIENT)
                return
        elif event.type == "model.finished":
            finish_reason = event.data.get("finish_reason")
            fail_info = map_finish_failure(finish_reason if isinstance(finish_reason, str) else None)
        elif event.type == "model.usage":
            usage = {
                "input_tokens": int(event.data.get("input_tokens") or 0),
                "output_tokens": int(event.data.get("output_tokens") or 0),
                "total_tokens": int(event.data.get("total_tokens") or 0),
            }
        # 不认识的模型事件直接丢掉，不能原样当成 RunEvent 推给浏览器。

    if state.cancel_event.is_set():
        await store.mark_cancelled(state, reason=CANCEL_REASON_CLIENT)
        return
    if fail_info is not None:
        # 已经推出去的 text.delta 留着。这里只补失败，不再写 run.completed。
        wrote_failure = await store.fail(state, fail_info)
        if not wrote_failure:
            await store.mark_cancelled(state, reason=CANCEL_REASON_CLIENT)
        return
    done = await store.complete(state, finish_reason=finish_reason, usage=usage)
    if not done:
        await store.mark_cancelled(state, reason=CANCEL_REASON_CLIENT)


async def _generate(
    store: RunStore,
    state: RunState,
    source: ModelStreamSource,
    request: ModelRequest,
    max_pre_text_attempts: int,
) -> None:
    """驱动一次 Run：先 run.started，再按首包策略消费模型流。"""

    started = await store.append(state, "run.started", {})
    if started is None:
        await store.mark_cancelled(state, reason=CANCEL_REASON_CLIENT)
        return

    attempt = 1
    while attempt <= max_pre_text_attempts:
        if state.cancel_event.is_set():
            await store.mark_cancelled(state, reason=CANCEL_REASON_CLIENT)
            return
        try:
            await _consume_once(store, state, source, request)
            return
        except asyncio.CancelledError:
            # 取消是控制信号，不能记成 run.failed，也不能在这里吞掉。
            raise
        except Exception as exc:
            payload = failure_payload(exc)
            # 只记类型名。异常消息可能含上游返回的请求信息。
            logger.warning(
                "run %s 模型流失败 code=%s exc=%s",
                state.run_id,
                payload.get("code"),
                type(exc).__name__,
            )
            if state.saw_text_delta:
                # 用户已经看见字了。再重试会造成两条不同的回答拼在同一次 Run 里。
                # 阶段改成 generate，表示失败发生在输出过程中，而不是还没连上。
                payload = {**payload, "stage": "generate"}
                wrote = await store.fail(state, payload)
                if not wrote:
                    await store.mark_cancelled(state, reason=CANCEL_REASON_CLIENT)
                return
            if payload.get("retryable") and attempt < max_pre_text_attempts:
                retry = await store.append(
                    state,
                    "run.retrying",
                    {"attempt": attempt, "code": payload["code"]},
                )
                if retry is None:
                    await store.mark_cancelled(state, reason=CANCEL_REASON_CLIENT)
                    return
                attempt += 1
                continue
            wrote = await store.fail(state, payload)
            if not wrote:
                await store.mark_cancelled(state, reason=CANCEL_REASON_CLIENT)
            return


async def run_producer(
    store: RunStore,
    state: RunState,
    source: ModelStreamSource,
    request: ModelRequest,
    *,
    max_pre_text_attempts: int = DEFAULT_MAX_PRE_TEXT_ATTEMPTS,
) -> None:
    """后台任务入口。CancelledError 会先落成 run.cancelled，然后继续抛出。

    task.cancel() 是协作式开关的兜底：生产者正堵在队列的 await 上时，
    只 set cancel_event 也要等下一条模型事件才看得见。cancel() 能立刻打断这个 await。
    """

    try:
        await _generate(store, state, source, request, max_pre_text_attempts)
    except asyncio.CancelledError:
        _clear_cancellation()
        try:
            await store.mark_cancelled(state, reason=CANCEL_REASON_CLIENT)
        except Exception as exc:
            logger.warning("写 run.cancelled 失败 exc=%s", type(exc).__name__)
        raise
    except Exception as exc:
        logger.warning("run %s 生产者未捕获异常 exc=%s", state.run_id, type(exc).__name__)
        wrote = await store.fail(state, failure_payload(exc))
        if not wrote:
            await store.mark_cancelled(state, reason=CANCEL_REASON_CLIENT)
