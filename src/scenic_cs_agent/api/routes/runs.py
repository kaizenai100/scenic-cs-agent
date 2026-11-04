"""创建 Run、订阅 SSE、显式取消。

三条路由故意拆开：
  POST /v1/runs                  开始后台生成，202 只返回 run_id
  GET  /v1/runs/{run_id}/events  订阅。断开连接只结束这一次订阅
  POST /v1/runs/{run_id}/cancel  取消。这是唯一会停掉生成任务的入口

断连不等于取消。用户刷新页面、代理掐掉空闲连接、读流的测试客户端提前关闭，
都只是「这个订阅者不看了」。生成任务继续往 Store 里写，下一次订阅带上
Last-Event-ID 就能从下一个 seq 接着看。要停生成，必须打取消接口。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from scenic_cs_agent.api.deps import get_heartbeat, get_store
from scenic_cs_agent.api.events import TERMINAL_EVENT_TYPES
from scenic_cs_agent.api.run_store import RunState, RunStore, caught_up_locked
from scenic_cs_agent.api.schemas import CancelRunResponse, CreateRunRequest, CreateRunResponse
from scenic_cs_agent.api.sse import SSE_MEDIA_TYPE, SSE_RESPONSE_HEADERS, encode_heartbeat, encode_sse

logger = logging.getLogger(__name__)

# 模型半天不吐字时，最多隔这么久发一帧注释，避免代理把连接当空闲关掉。
DEFAULT_HEARTBEAT_INTERVAL = 15.0

# 由启动装配或测试放进 app.state。传入这一次 Run 的状态和请求正文。
StartRun = Callable[[RunState, CreateRunRequest], Awaitable[None]]


def build_runs_router() -> APIRouter:
    """每次建应用都新做一个路由，避免同一个路由器挂到两个应用上。"""

    router = APIRouter()

    @router.post("/v1/runs", status_code=202, response_model=CreateRunResponse)
    async def create_run(
        body: CreateRunRequest,
        request: Request,
        store: RunStore = Depends(get_store),
    ) -> CreateRunResponse:
        """启动后台生成。202 表示 Run 已接受，不表示模型已经开始吐字。"""

        start = getattr(request.app.state, "start_run", None)
        if not callable(start):
            raise HTTPException(status_code=503, detail="服务还没装配完成")
        state = store.create()
        state.task = asyncio.create_task(start(state, body), name=f"run-{state.run_id}")
        state.task.add_done_callback(_log_producer_error)
        return CreateRunResponse(run_id=state.run_id)

    @router.get("/v1/runs/{run_id}/events")
    async def stream_events(
        run_id: str,
        request: Request,
        store: RunStore = Depends(get_store),
        heartbeat_interval: float = Depends(get_heartbeat),
    ) -> StreamingResponse:
        """订阅事件流。Last-Event-ID（或查询参数）之前的 seq 不再发送。"""

        state = store.get(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail="run 不存在")
        last_seq = _last_seq_from_request(request)
        return StreamingResponse(
            _event_stream(state, request, last_seq, heartbeat_interval),
            media_type=SSE_MEDIA_TYPE,
            headers=SSE_RESPONSE_HEADERS,
        )

    @router.post("/v1/runs/{run_id}/cancel", response_model=CancelRunResponse)
    async def cancel_run(run_id: str, store: RunStore = Depends(get_store)) -> CancelRunResponse:
        """显式取消。断连不会走到这里。

        先比较并交换到 cancelling，再 cancel 任务。顺序不能反：
        如果先 cancel 任务，生产者可能在状态还是 running 时就写 run.cancelled，
        和正在进行的 complete() 抢终态。
        """

        state = store.get(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail="run 不存在")
        accepted = await store.request_cancel(state)
        task = state.task
        if accepted and task is not None and not task.done():
            task.cancel()
        return CancelRunResponse(run_id=state.run_id, status=state.status.value, accepted=accepted)

    return router


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
    ?last_event_id= 或 ?Last-Event-ID= 。两边都有时以头为准。
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
    """

    cursor = last_seq
    try:
        while True:
            if await request.is_disconnected():
                return

            async with state.changed:
                fresh = [event for event in state.events if event.seq > cursor]
                if not fresh:
                    if caught_up_locked(state, cursor):
                        return
                    try:
                        await asyncio.wait_for(state.changed.wait(), heartbeat_interval)
                    except TimeoutError:
                        should_heartbeat = True
                    else:
                        continue
                else:
                    should_heartbeat = False

            if should_heartbeat:
                yield encode_heartbeat()
                continue

            for event in fresh:
                cursor = event.seq
                yield encode_sse(event)
                if event.type in TERMINAL_EVENT_TYPES:
                    return
                if await request.is_disconnected():
                    return
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("sse 订阅异常，只结束订阅 exc=%s", type(exc).__name__)
        return


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
