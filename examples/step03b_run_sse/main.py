# Run 级 SSE 演示（默认离线，不读密钥、不访问模型）
#
# 跑：
#   uv run python examples/step03b_run_sse/main.py
#
# 演示两件事：
#   1. 创建 Run，第一路订阅读到两条正文后主动断开（不断 Run），
#      第二路带 Last-Event-ID，只打印还没看过的 seq，直到 run.completed。
#   2. 另开一条 Run，收到第一条 text.delta 后调用取消接口，打印 run.cancelled。
#
# 可选：
#   SERVE=1 uv run python examples/step03b_run_sse/main.py
#       挂到 127.0.0.1:8765（端口可用 PORT 改）。自己用 curl 订阅：
#       curl -N -X POST http://127.0.0.1:8765/v1/runs \
#         -H 'content-type: application/json' -d '{"message":"你好"}'
#       curl -N http://127.0.0.1:8765/v1/runs/<run_id>/events
#       curl -N -H 'Last-Event-ID: 2' http://127.0.0.1:8765/v1/runs/<run_id>/events
#       curl -X POST http://127.0.0.1:8765/v1/runs/<run_id>/cancel
#   SCENIC_SSE_USE_DEEPSEEK=1 uv run python examples/step03b_run_sse/main.py
#       改用 DeepSeekChatAdapter.stream。需要 DEEPSEEK_API_KEY，经 get_config 读取。
#
# 心跳是 `: keep-alive` 注释帧，不占 seq，默认 15 秒才发一帧，这条短演示里通常看不到。

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator

import httpx
from httpx import ASGITransport

from scenic_cs_agent.api.app import create_app


def _format(payload: dict) -> str:
    """一行：序号、类型、以及和这个类型有关的短文本。"""

    data = payload.get("data") or {}
    if payload["type"] == "text.delta":
        extra = data.get("delta", "")
    elif payload["type"] == "run.cancelled":
        extra = data.get("partial_text", "")
    elif payload["type"] == "run.failed":
        extra = data.get("hint", "")
    elif payload["type"] == "message.completed":
        extra = data.get("text", "")
    else:
        extra = ""
    text = extra if isinstance(extra, str) else ""
    return f"[{payload['seq']}] {payload['type']} {text}".rstrip()


async def _iter_sse(
    app,
    path: str,
    *,
    last_event_id: int | None = None,
    stop_after_deltas: int | None = None,
) -> AsyncIterator[dict]:
    """增量读一条 SSE。stop_after_deltas 到了就让服务端看到 http.disconnect。

    断开只结束这一次订阅。生成任务不在这里取消。
    spec_version 2.5：避免服务器再开一条 receive() 把断连信号抢走。
    """

    disconnect = asyncio.Event()
    chunks: asyncio.Queue[bytes | None] = asyncio.Queue()
    headers: list[tuple[bytes, bytes]] = []
    if last_event_id is not None:
        headers.append((b"last-event-id", str(last_event_id).encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.5"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("test", 80),
    }
    request_sent = False

    async def receive() -> dict:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        if disconnect.is_set():
            return {"type": "http.disconnect"}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        if message["type"] != "http.response.body":
            return
        body = message.get("body", b"")
        if body:
            await chunks.put(body)
        if not message.get("more_body", False):
            await chunks.put(None)

    app_task = asyncio.create_task(app(scope, receive, send))
    buffer = ""
    deltas = 0
    try:
        while True:
            piece = await chunks.get()
            if piece is None:
                break
            buffer += piece.decode()
            while "\n\n" in buffer:
                block, buffer = buffer.split("\n\n", 1)
                if block.startswith(":"):
                    print("(heartbeat)", flush=True)
                    continue
                payload = json.loads(block.split("data: ", 1)[1])
                yield payload
                if payload["type"] == "text.delta":
                    deltas += 1
                if stop_after_deltas is not None and deltas >= stop_after_deltas:
                    # 订阅者走了。下一次 is_disconnected() 为真，生成继续跑。
                    disconnect.set()
                    stop_after_deltas = None
        await app_task
    finally:
        disconnect.set()
        if not app_task.done():
            app_task.cancel()
            try:
                await app_task
            except asyncio.CancelledError:
                pass


async def _create(client: httpx.AsyncClient, message: str) -> str:
    response = await client.post("/v1/runs", json={"message": message})
    response.raise_for_status()
    return response.json()["run_id"]


async def _show_replay(app) -> None:
    """第一路读两条正文就断开，第二路从 Last-Event-ID 接着看。"""

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://scenic", timeout=30) as client:
        run_id = await _create(client, "明天想去苏州，门票多少钱？")
    print(f"\n== 重放 demo  run_id={run_id} ==", flush=True)
    print("-- 连接 1：读到两条正文后断开，不取消 Run --", flush=True)
    last_seq = 0
    async for payload in _iter_sse(app, f"/v1/runs/{run_id}/events", stop_after_deltas=2):
        print(_format(payload), flush=True)
        last_seq = payload["seq"]
    print(f"-- 连接 1 结束，最后处理的 seq={last_seq} --", flush=True)
    print(f"-- 连接 2：Last-Event-ID: {last_seq} --", flush=True)
    types: list[str] = []
    async for payload in _iter_sse(
        app,
        f"/v1/runs/{run_id}/events",
        last_event_id=last_seq,
    ):
        print(_format(payload), flush=True)
        types.append(payload["type"])
        if payload["seq"] <= last_seq:
            raise SystemExit(f"重放不该再出现 seq={payload['seq']}")
    if "run.completed" not in types:
        raise SystemExit("第二路没有看到 run.completed")


async def _show_cancel(app) -> None:
    """收到第一条正文就打取消接口，流里应出现 run.cancelled 而不是 run.completed。"""

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://scenic", timeout=30) as client:
        run_id = await _create(client, "只看一半就取消")
        print(f"\n== 取消 demo  run_id={run_id} ==", flush=True)
        cancelled = False
        types: list[str] = []
        async for payload in _iter_sse(app, f"/v1/runs/{run_id}/events"):
            print(_format(payload), flush=True)
            types.append(payload["type"])
            if payload["type"] == "text.delta" and not cancelled:
                cancelled = True
                response = await client.post(f"/v1/runs/{run_id}/cancel")
                print(f"POST /cancel -> {response.status_code} {response.json()}", flush=True)
    if "run.cancelled" not in types:
        raise SystemExit("取消后没有看到 run.cancelled")
    if "run.completed" in types:
        raise SystemExit("取消后不应该再 run.completed")


async def _demo() -> None:
    app = create_app()
    await _show_replay(app)
    await _show_cancel(app)
    print("\n完成。", flush=True)


def main() -> None:
    if os.getenv("SERVE", "").strip() == "1":
        import uvicorn

        port = int(os.getenv("PORT", "8765"))
        uvicorn.run(create_app(), host="127.0.0.1", port=port)
        return
    asyncio.run(_demo())


if __name__ == "__main__":
    main()
