"""FastAPI：创建 Run、订阅 SSE、显式取消。

三条路由故意拆开：
  POST /v1/runs                  开始后台生成，202 只返回 run_id
  GET  /v1/runs/{run_id}/events  订阅。断开连接只结束这一次订阅
  POST /v1/runs/{run_id}/cancel  取消。这是唯一会停掉生成任务的入口

断连不等于取消。用户刷新页面、代理掐掉空闲连接、读流的测试客户端提前关闭，
都只是「这个订阅者不看了」。生成任务继续往 Store 里写，下一次订阅带上
Last-Event-ID 就能从下一个 seq 接着看。要停生成，必须打取消接口：
先把状态从 running 收成 cancelling（协作式 cancel_event），再 task.cancel()
打断正堵在 await 上的生产者。生产者负责写下 run.cancelled，不在这里写，
否则 partial_text 可能还不是最终值。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from scenic_cs_agent.api.demo_runner import DEFAULT_SYSTEM, build_demo_source
from scenic_cs_agent.api.events import TERMINAL_EVENT_TYPES
from scenic_cs_agent.api.producer import ModelStreamSource, run_producer
from scenic_cs_agent.api.run_store import RunState, RunStore, caught_up_locked
from scenic_cs_agent.api.sse import SSE_MEDIA_TYPE, SSE_RESPONSE_HEADERS, encode_heartbeat, encode_sse
from scenic_cs_agent.runtime.model_types import ModelRequest

logger = logging.getLogger(__name__)

# 模型半天不吐字时，最多隔这么久发一帧注释，避免代理把连接当空闲关掉。
DEFAULT_HEARTBEAT_INTERVAL = 15.0


class CreateRunRequest(BaseModel):
    """创建 Run 的正文。system 可选，不传就用售前助手的短边界。"""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000)
    system: str | None = Field(default=None, max_length=8000)


class CreateRunResponse(BaseModel):
    """202 的正文。只有 run_id，没有模型输出。输出在 SSE 里。"""

    run_id: str


class CancelRunResponse(BaseModel):
    """取消接口的应答。

    accepted 为真表示这次调用把 running 推进到了 cancelling。
    已经结束的 Run 再取消，accepted 为假，status 保持原来的终态。
    """

    run_id: str
    status: str
    accepted: bool


def _parse_last_seq(raw: str | None) -> int:
    """Last-Event-ID 或查询参数 → 已经确认收过的 seq。缺省表示从头看。

    非法值直接 400。如果当成 0，客户端会以为自己续上了，其实又从头收了一遍。
    """

    if raw is None or raw.strip() == "":
        return 0
    try:
        value = int(raw.strip())
    except ValueError:
        raise HTTPException(status_code=400, detail="Last-Event-ID 必须是十进制整数") from None
    if value < 0:
        raise HTTPException(status_code=400, detail="Last-Event-ID 不能为负")
    return value


def _last_seq_from_request(request: Request) -> int:
    """优先采用标准头 Last-Event-ID，否则看查询参数。

    EventSource 重连只会自动带头，带不上自定义头的客户端可以用
    ?last_event_id= 或 ?Last-Event-ID= 。两边都有时以头为准，
    因为那是协议里「我已经处理到这里」的原话。
    """

    header = request.headers.get("last-event-id")
    if header is not None:
        return _parse_last_seq(header)
    query = request.query_params.get("last_event_id")
    if query is None:
        query = request.query_params.get("Last-Event-ID")
    return _parse_last_seq(query)


async def _event_stream(
    state: RunState,
    request: Request,
    last_seq: int,
    heartbeat_interval: float,
) -> AsyncIterator[str]:
    """把 Store 里 seq 大于 last_seq 的事件编成 SSE，没有新事件就发心跳。

    每一轮先问 request.is_disconnected()。为真就 return：
    不 set cancel_event，也不 task.cancel()。生成任务和这次响应不是父子任务，
    响应结束不会把 Run 一起取消。

    框架自己的断连监听也可能把本协程取消掉。CancelledError 照样抛出，
    但仍然不去动 Run。取消 Run 只有取消接口那一条路。
    """

    cursor = last_seq
    try:
        while True:
            # 客户端已经走了。停掉的是订阅，不是 Run。
            if await request.is_disconnected():
                return

            async with state.changed:
                fresh = [event for event in state.events if event.seq > cursor]
                if not fresh:
                    # 已经是终态且序号都发过了：直接结束，不要再心跳。
                    # 用不加锁的判断，因为这把 Condition 锁不可重入。
                    if caught_up_locked(state, cursor):
                        return
                    try:
                        # 等到生产者 notify，或者心跳期限到了。
                        # 超时不是错误：这条连接需要一帧注释才能活过代理的空闲切断。
                        await asyncio.wait_for(state.changed.wait(), heartbeat_interval)
                    except TimeoutError:
                        should_heartbeat = True
                    else:
                        # 被叫醒了。回到循环顶部重新拷贝事件，避免漏掉刚写进去的那条。
                        continue
                else:
                    should_heartbeat = False

            if should_heartbeat:
                # 心跳不进 Store，不增加 next_seq。必须在锁外 yield，
                # 否则慢客户端会把生产者和别的订阅者一起堵住。
                yield encode_heartbeat()
                continue

            for event in fresh:
                cursor = event.seq
                yield encode_sse(event)
                if event.type in TERMINAL_EVENT_TYPES:
                    return
                # 一批里也可能已经断线。不再把后面的帧写进一个已经没人读的套接字。
                if await request.is_disconnected():
                    return
    except asyncio.CancelledError:
        # 响应任务被取消（常见于服务器自己的断连监听）。
        # 这里重新抛出，只结束订阅；生产者是独立 task，不会跟着死。
        raise
    except Exception as exc:
        # 响应头一旦送出，再把异常抛出生成器，客户端只能看到连接被掐断。
        # 模型失败应已经写成 run.failed；这里只把坏掉的订阅停掉，不取消 Run。
        logger.warning("sse 订阅异常，只结束订阅 exc=%s", type(exc).__name__)
        return


def create_app(
    *,
    source_factory: Callable[[], ModelStreamSource] | None = None,
    heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    store: RunStore | None = None,
    max_pre_text_attempts: int = 2,
) -> FastAPI:
    """组装应用。测试传入 source_factory 和更短的心跳，示例用默认的脚本模型。

    store 挂在 app.state 上，测试可以在不走 HTTP 的情况下核对状态。
    这不是给外部客户端的接口。
    """

    if heartbeat_interval <= 0:
        raise ValueError("heartbeat_interval 必须大于 0")

    app = FastAPI(title="scenic-cs-agent", version="0.1.0")
    run_store = store or RunStore()
    factory = source_factory or build_demo_source
    app.state.store = run_store

    @app.post("/v1/runs", status_code=202, response_model=CreateRunResponse)
    async def create_run(body: CreateRunRequest) -> CreateRunResponse:
        """启动后台生成。202 表示 Run 已接受，不表示模型已经开始吐字。"""

        state = run_store.create()
        model_request = ModelRequest(
            system=body.system or DEFAULT_SYSTEM,
            user=body.message,
            max_output_tokens=512,
        )
        # 每个 Run 一份新的 source。脚本模型把列表迭代一次就耗尽了，不能跨 Run 复用。
        source = factory()
        state.task = asyncio.create_task(
            run_producer(
                run_store,
                state,
                source,
                model_request,
                max_pre_text_attempts=max_pre_text_attempts,
            ),
            name=f"run-{state.run_id}",
        )
        state.task.add_done_callback(_log_producer_error)
        return CreateRunResponse(run_id=state.run_id)

    @app.get("/v1/runs/{run_id}/events")
    async def stream_events(run_id: str, request: Request) -> StreamingResponse:
        """订阅事件流。Last-Event-ID（或查询参数）之前的 seq 不再发送。"""

        state = run_store.get(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail="run 不存在")
        last_seq = _last_seq_from_request(request)
        return StreamingResponse(
            _event_stream(state, request, last_seq, heartbeat_interval),
            media_type=SSE_MEDIA_TYPE,
            headers=SSE_RESPONSE_HEADERS,
        )

    @app.post("/v1/runs/{run_id}/cancel", response_model=CancelRunResponse)
    async def cancel_run(run_id: str) -> CancelRunResponse:
        """显式取消。断连不会走到这里。

        先比较并交换到 cancelling，再 cancel 任务。顺序不能反：
        如果先 cancel 任务，生产者可能在状态还是 running 时就写 run.cancelled，
        和正在进行的 complete() 抢终态。request_cancel 失败就说明已经收尾了，
        不要再 cancel 一次，否则会把取消计数加到一个正在清理的任务上。
        """

        state = run_store.get(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail="run 不存在")
        accepted = await run_store.request_cancel(state)
        task = state.task
        if accepted and task is not None and not task.done():
            task.cancel()
        return CancelRunResponse(run_id=state.run_id, status=state.status.value, accepted=accepted)

    return app


def _log_producer_error(task: asyncio.Task[None]) -> None:
    """后台任务如果真的漏了异常，至少在日志里留类型名。

    正常的取消和已经写成 run.failed 的失败不会进这里。
    不打印异常消息，避免日志把上游文本再扩散一遍。
    """

    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("run 生产者异常退出 exc=%s", type(exc).__name__)
