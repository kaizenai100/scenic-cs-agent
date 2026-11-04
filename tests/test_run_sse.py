"""Run 级 SSE 的离线测试。

不访问模型供应商。生产者全部用脚本模型或会抛异常的假模型。
httpx.ASGITransport 把整段响应收齐，适合「一直读到结束」的用例。
它要等响应结束才把 http.disconnect 交给应用，测不了「读到一半客户端走了」，
所以断连用例直接按 ASGI 调应用，并在下一次 receive() 返回 http.disconnect。
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from httpx import ASGITransport

from scenic_cs_agent.bootstrap import create_app
from scripted_model import ScriptedModel
from scenic_cs_agent.api.events import tool_completed_data, tool_started_data
from scenic_cs_agent.api.producer import StreamFailure
from scenic_cs_agent.api.run_store import RunStatus, RunStore
from scenic_cs_agent.api.sse import encode_heartbeat, encode_sse
from scenic_cs_agent.runtime.model_types import ModelStreamEvent


def _event(seq: int, event_type: str, **data: object) -> object:
    """凑一条只用于编码测试的 RunEvent。不经过 Store，序号自己指定。"""

    from scenic_cs_agent.api.events import RunEvent

    return RunEvent(
        schema_version="1",
        run_id="run_test",
        seq=seq,
        type=event_type,
        created_at="2026-01-01T00:00:00+00:00",
        data=dict(data),
    )


def parse_sse(raw: str) -> tuple[list[dict], int]:
    """把一整段 SSE 正文拆成 RunEvent 字典，并数心跳帧。"""

    events: list[dict] = []
    heartbeats = 0
    for block in raw.split("\n\n"):
        if block == "":
            continue
        if block.startswith(":"):
            assert block == ": keep-alive", block
            heartbeats += 1
            continue
        event_name = None
        event_id = None
        data_line = None
        for line in block.split("\n"):
            # data 必须只有一行。出现第二行说明 JSON 被换行拆开了。
            assert not line.startswith("data:") or data_line is None
            if line.startswith("id: "):
                event_id = line[len("id: ") :]
            elif line.startswith("event: "):
                event_name = line[len("event: ") :]
            elif line.startswith("data: "):
                data_line = line[len("data: ") :]
            else:
                raise AssertionError(f"无法识别的 SSE 行: {line!r}")
        assert data_line is not None
        assert "\n" not in data_line
        payload = json.loads(data_line)
        assert event_id == str(payload["seq"])
        assert event_name == payload["type"]
        events.append(payload)
    return events, heartbeats


def _script(*items: ModelStreamEvent | BaseException, delay: float = 0.0) -> ScriptedModel:
    return ScriptedModel(list(items), delay=delay)


def _ok_script(text: str = "你好") -> ScriptedModel:
    return _script(
        ModelStreamEvent(type="text.delta", data={"delta": text}),
        ModelStreamEvent(type="model.finished", data={"finish_reason": "stop"}),
        ModelStreamEvent(
            type="model.usage",
            data={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        ),
    )


async def _settle(app, run_id: str) -> None:
    """等后台生产者退出，避免测试结束时还挂着任务。"""

    state = app.state.store.get(run_id)
    assert state is not None and state.task is not None
    try:
        await asyncio.wait_for(state.task, timeout=2)
    except asyncio.CancelledError:
        pass


async def _post_run(client: httpx.AsyncClient, message: str = "想去苏州") -> str:
    response = await client.post("/v1/runs", json={"message": message})
    assert response.status_code == 202, response.text
    body = response.json()
    assert set(body) == {"run_id"}
    assert body["run_id"].startswith("run_")
    return body["run_id"]


async def _read_sse(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str] | None = None,
) -> tuple[int, httpx.Headers, str]:
    async with client.stream("GET", url, headers=headers) as response:
        status = response.status_code
        response_headers = response.headers
        body = (await response.aread()).decode()
    return status, response_headers, body


def test_encode_sse_frame_and_heartbeat() -> None:
    """id 是 seq，event 是 type，data 是单行 JSON。心跳是不带 id 的注释帧。"""

    frame = encode_sse(_event(3, "text.delta", delta="你\n好"))
    assert frame.endswith("\n\n")
    lines = frame.split("\n")
    assert lines[0] == "id: 3"
    assert lines[1] == "event: text.delta"
    assert lines[2].startswith("data: ")
    assert "\n" not in lines[2]
    # 正文里的换行只能以 JSON 转义出现，否则这一帧会被拆成两帧。
    assert "\\n" in lines[2]
    payload = json.loads(lines[2].removeprefix("data: "))
    assert payload == {
        "schema_version": "1",
        "run_id": "run_test",
        "seq": 3,
        "type": "text.delta",
        "created_at": "2026-01-01T00:00:00+00:00",
        "data": {"delta": "你\n好"},
    }
    assert encode_heartbeat() == ": keep-alive\n\n"
    assert "id:" not in encode_heartbeat()
    assert "data:" not in encode_heartbeat()


async def test_create_run_streams_until_completed() -> None:
    """创建返回 202。订阅一直读到 run.completed，序号连续，反缓冲头齐全。"""

    app = create_app(source_factory=_ok_script, heartbeat_interval=15)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=3) as client:
        run_id = await _post_run(client)
        status, headers, raw = await _read_sse(client, f"/v1/runs/{run_id}/events")
    assert status == 200
    assert headers["content-type"].startswith("text/event-stream")
    assert headers["cache-control"] == "no-cache, no-transform"
    assert headers["x-accel-buffering"] == "no"
    assert headers["connection"] == "keep-alive"

    events, _heartbeats = parse_sse(raw)
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    assert [event["type"] for event in events] == [
        "run.started",
        "text.delta",
        "message.completed",
        "run.completed",
    ]
    assert events[1]["data"] == {"delta": "你好"}
    assert events[2]["data"]["text"] == "你好"
    assert events[2]["data"]["finish_reason"] == "stop"
    assert events[3]["data"]["usage"] == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
    }
    assert all(event["run_id"] == run_id for event in events)
    await _settle(app, run_id)
    assert app.state.store.get(run_id).status == RunStatus.COMPLETED


async def test_heartbeat_does_not_consume_seq() -> None:
    """模型暂时不吐字时发注释帧，下一条业务事件的 seq 仍然紧挨着上一条。"""

    app = create_app(
        source_factory=lambda: _script(
            ModelStreamEvent(type="text.delta", data={"delta": "甲"}),
            ModelStreamEvent(type="model.finished", data={"finish_reason": "stop"}),
            delay=0.12,
        ),
        heartbeat_interval=0.04,
    )
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=3) as client:
        run_id = await _post_run(client)
        _status, _headers, raw = await _read_sse(client, f"/v1/runs/{run_id}/events")
    events, heartbeats = parse_sse(raw)
    assert heartbeats >= 1
    assert ": keep-alive" in raw
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    assert "heartbeat" not in {event["type"] for event in events}
    await _settle(app, run_id)


async def test_last_event_id_replays_only_the_tail() -> None:
    """头和查询参数都表示「这个 seq 我已经处理过了」，只重放更大的 seq。"""

    app = create_app(source_factory=_ok_script, heartbeat_interval=15)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=3) as client:
        run_id = await _post_run(client)
        _status, _headers, raw = await _read_sse(client, f"/v1/runs/{run_id}/events")
        events, _ = parse_sse(raw)
        assert events[-1]["type"] == "run.completed"
        last_seq = events[-1]["seq"]

        status, _headers, by_header = await _read_sse(
            client,
            f"/v1/runs/{run_id}/events?last_event_id=0",
            headers={"Last-Event-ID": "2"},
        )
        assert status == 200
        replayed, _ = parse_sse(by_header)
        assert [event["seq"] for event in replayed] == list(range(3, last_seq + 1))

        status, _headers, by_query = await _read_sse(
            client,
            f"/v1/runs/{run_id}/events?Last-Event-ID=1",
        )
        assert status == 200
        queried, _ = parse_sse(by_query)
        assert queried[0]["seq"] == 2

        status, _headers, done = await _read_sse(
            client,
            f"/v1/runs/{run_id}/events",
            headers={"Last-Event-ID": str(last_seq)},
        )
        assert status == 200
        assert done == ""

        bad = await client.get(f"/v1/runs/{run_id}/events", headers={"Last-Event-ID": "abc"})
        assert bad.status_code == 400
    await _settle(app, run_id)


async def test_disconnect_does_not_cancel_and_replay_continues() -> None:
    """第一路连接在正文中途断开。Run 继续，第二路用 Last-Event-ID 看到后半段。"""

    app = create_app(
        source_factory=lambda: _script(
            ModelStreamEvent(type="text.delta", data={"delta": "甲"}),
            ModelStreamEvent(type="text.delta", data={"delta": "乙"}),
            ModelStreamEvent(type="model.finished", data={"finish_reason": "stop"}),
            delay=0.2,
        ),
        heartbeat_interval=0.05,
    )
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=3) as client:
        run_id = await _post_run(client)

    seen_delta = False

    def stop_after_first_delta(payload: dict) -> bool:
        nonlocal seen_delta
        if payload["type"] == "text.delta":
            seen_delta = True
            return True
        return False

    first = await _drive_sse(
        app,
        f"/v1/runs/{run_id}/events",
        stop_when=stop_after_first_delta,
    )
    assert seen_delta
    first_events = [frame for frame in first if "type" in frame]
    assert any(event["data"].get("delta") == "甲" for event in first_events)
    assert all(event["data"].get("delta") != "乙" for event in first_events)
    assert all(event["type"] != "run.completed" for event in first_events)
    assert all(event["type"] != "run.cancelled" for event in first_events)
    cursor = first_events[-1]["seq"]

    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=3) as client:
        status, _headers, raw = await _read_sse(
            client,
            f"/v1/runs/{run_id}/events",
            headers={"Last-Event-ID": str(cursor)},
        )
    assert status == 200
    replayed, _ = parse_sse(raw)
    assert replayed
    assert all(event["seq"] > cursor for event in replayed)
    assert any(event["data"].get("delta") == "乙" for event in replayed)
    assert replayed[-1]["type"] == "run.completed"
    await _settle(app, run_id)
    state = app.state.store.get(run_id)
    assert state.status == RunStatus.COMPLETED
    assert all(event.type != "run.cancelled" for event in state.events)


async def test_cancel_endpoint_emits_run_cancelled_with_partial_text() -> None:
    """取消接口把 Run 停住。断连不会产生 run.cancelled，这个接口会。"""

    app = create_app(
        source_factory=lambda: _script(
            ModelStreamEvent(type="text.delta", data={"delta": "已写"}),
            ModelStreamEvent(type="text.delta", data={"delta": "不应出现"}),
            ModelStreamEvent(type="model.finished", data={"finish_reason": "stop"}),
            delay=0.3,
        ),
        heartbeat_interval=15,
    )
    transport = ASGITransport(app=app)

    async def read_until_end(client: httpx.AsyncClient, run_id: str) -> str:
        _status, _headers, raw = await _read_sse(client, f"/v1/runs/{run_id}/events")
        return raw

    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=3) as client:
        run_id = await _post_run(client)
        reader = asyncio.create_task(read_until_end(client, run_id))
        state = app.state.store.get(run_id)
        for _ in range(50):
            if any(event.type == "text.delta" for event in state.events):
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("生产者没有写出第一条正文")

        cancel = await client.post(f"/v1/runs/{run_id}/cancel")
        assert cancel.status_code == 200
        body = cancel.json()
        assert body["accepted"] is True
        assert body["status"] in {"cancelling", "cancelled"}
        raw = await asyncio.wait_for(reader, timeout=2)

        again = await client.post(f"/v1/runs/{run_id}/cancel")
        assert again.status_code == 200
        assert again.json()["accepted"] is False

    events, _ = parse_sse(raw)
    assert events[-1]["type"] == "run.cancelled"
    assert events[-1]["data"]["reason"] == "client_cancel"
    assert events[-1]["data"]["partial_text"] == "已写"
    assert all(event["data"].get("delta") != "不应出现" for event in events)
    assert all(event["type"] != "run.completed" for event in events)
    assert sum(event["type"] == "run.cancelled" for event in events) == 1
    await _settle(app, run_id)
    assert app.state.store.get(run_id).status == RunStatus.CANCELLED


async def test_failure_after_first_delta_stays_in_the_stream() -> None:
    """第一条正文已经可见之后，异常必须变成 run.failed，响应仍然是 200。"""

    secret = "sk-LIVE-SECRET-VALUE"
    app = create_app(
        source_factory=lambda: _script(
            ModelStreamEvent(type="text.delta", data={"delta": "部分"}),
            RuntimeError(f"上游炸了 {secret} prompt=系统提示词全文"),
        ),
        heartbeat_interval=15,
    )
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=3) as client:
        run_id = await _post_run(client)
        status, _headers, raw = await _read_sse(client, f"/v1/runs/{run_id}/events")
    assert status == 200
    assert secret not in raw
    assert "系统提示词" not in raw
    assert "Traceback" not in raw
    events, _ = parse_sse(raw)
    assert [event["type"] for event in events] == ["run.started", "text.delta", "run.failed"]
    failure = events[-1]["data"]
    assert set(failure) == {"code", "stage", "retryable", "hint"}
    assert failure["code"] == "internal"
    assert failure["stage"] == "generate"
    assert failure["retryable"] is False
    assert failure["hint"] == "生成失败，请稍后重试"
    await _settle(app, run_id)
    assert app.state.store.get(run_id).status == RunStatus.FAILED


async def test_retryable_failure_before_first_delta_retries_once() -> None:
    """还没有任何 text.delta 时，可重试错误先写 run.retrying，再换一次成功结果。"""

    class Flaky:
        def __init__(self) -> None:
            self.calls = 0

        def stream(self, request: object):
            del request
            self.calls += 1
            if self.calls == 1:
                raise StreamFailure(
                    "rate_limit",
                    stage="connect",
                    retryable=True,
                    hint="服务繁忙，请稍后重试",
                )
            yield ModelStreamEvent(type="text.delta", data={"delta": "好"})
            yield ModelStreamEvent(type="model.finished", data={"finish_reason": "stop"})

    model = Flaky()
    app = create_app(source_factory=lambda: model, heartbeat_interval=15)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=3) as client:
        run_id = await _post_run(client)
        status, _headers, raw = await _read_sse(client, f"/v1/runs/{run_id}/events")
    assert status == 200
    events, _ = parse_sse(raw)
    assert [event["type"] for event in events] == [
        "run.started",
        "run.retrying",
        "text.delta",
        "message.completed",
        "run.completed",
    ]
    assert events[1]["data"] == {"attempt": 1, "code": "rate_limit"}
    assert model.calls == 2
    await _settle(app, run_id)


async def test_non_retryable_failure_before_text_does_not_retry() -> None:
    """认证 / 参数这类错误不在重试白名单里，第一条正文前也直接 run.failed。"""

    class Reject:
        def __init__(self) -> None:
            self.calls = 0

        def stream(self, request: object):
            del request
            self.calls += 1
            raise StreamFailure(
                "rejected",
                stage="connect",
                retryable=False,
                hint="请求被拒绝",
            )
            yield  # pragma: no cover - 让函数成为生成器

    model = Reject()
    app = create_app(source_factory=lambda: model, heartbeat_interval=15)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=3) as client:
        run_id = await _post_run(client)
        _status, _headers, raw = await _read_sse(client, f"/v1/runs/{run_id}/events")
    events, _ = parse_sse(raw)
    assert [event["type"] for event in events] == ["run.started", "run.failed"]
    assert events[-1]["data"]["retryable"] is False
    assert model.calls == 1
    await _settle(app, run_id)


@pytest.mark.parametrize(
    ("finish_reason", "code"),
    [
        ("length", "truncated"),
        ("content_filter", "refusal"),
        ("tool_calls", "unsupported"),
    ],
)
async def test_bad_finish_reason_is_run_failed(finish_reason: str, code: str) -> None:
    """模型说结束了，不代表这次 Run 成功。截断、拒绝、未执行的工具调用都是失败。"""

    app = create_app(
        source_factory=lambda: _script(
            ModelStreamEvent(type="text.delta", data={"delta": "半"}),
            ModelStreamEvent(type="model.finished", data={"finish_reason": finish_reason}),
        ),
        heartbeat_interval=15,
    )
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=3) as client:
        run_id = await _post_run(client)
        _status, _headers, raw = await _read_sse(client, f"/v1/runs/{run_id}/events")
    events, _ = parse_sse(raw)
    assert events[-1]["type"] == "run.failed"
    assert events[-1]["data"]["code"] == code
    assert all(event["type"] != "run.completed" for event in events)
    await _settle(app, run_id)


async def test_unknown_run_and_empty_message() -> None:
    app = create_app(source_factory=_ok_script)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=3) as client:
        missing = await client.get("/v1/runs/run_missing/events")
        assert missing.status_code == 404
        missing_cancel = await client.post("/v1/runs/run_missing/cancel")
        assert missing_cancel.status_code == 404
        empty = await client.post("/v1/runs", json={"message": ""})
        assert empty.status_code == 422


async def test_tool_events_and_cancel_cas() -> None:
    """工具事件先把形状留好。取消状态只能从 running 走进 cancelling 一次。"""

    store = RunStore()
    state = store.create()
    started = await store.append(state, "tool.started", tool_started_data("call_1", "lookup_ticket"))
    finished = await store.append(
        state,
        "tool.completed",
        tool_completed_data("call_1", "lookup_ticket", ok=True),
    )
    assert started is not None and started.seq == 1
    assert finished is not None and finished.data == {
        "tool_call_id": "call_1",
        "name": "lookup_ticket",
        "ok": True,
    }

    assert await store.request_cancel(state) is True
    assert state.status == RunStatus.CANCELLING
    assert state.cancel_event.is_set()
    assert await store.request_cancel(state) is False
    assert await store.complete(state, finish_reason="stop", usage=None) is False
    assert await store.fail(state, {"code": "internal", "stage": "generate", "retryable": False, "hint": "不会写出"}) is False
    assert await store.mark_cancelled(state, reason="client_cancel") is True
    assert await store.mark_cancelled(state, reason="client_cancel") is False
    assert [event.type for event in state.events] == [
        "tool.started",
        "tool.completed",
        "run.cancelled",
    ]
    assert state.events[-1].data["partial_text"] == ""


async def _drive_sse(app, path: str, *, stop_when) -> list[dict]:
    """按 ASGI 读 SSE。stop_when 返回真时，下一次 receive() 给出 http.disconnect。

    spec_version 用 2.5，这样服务器不会再开一条并行的 receive() 去听断连，
    断连信号只会落到我们自己的 is_disconnected() 上。
    """

    disconnect = asyncio.Event()
    chunks: asyncio.Queue[bytes | None] = asyncio.Queue()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.5"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("test", 80),
    }
    request_sent = False

    async def receive() -> dict:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        # 已经要求断连时立刻返回。is_disconnected 不会傻等，消息必须已经在手上。
        if disconnect.is_set():
            return {"type": "http.disconnect"}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        if message["type"] == "http.response.body":
            body = message.get("body", b"")
            if body:
                await chunks.put(body)
            if not message.get("more_body", False):
                await chunks.put(None)

    app_task = asyncio.create_task(app(scope, receive, send))
    frames: list[dict] = []
    buffer = ""
    try:
        while True:
            piece = await asyncio.wait_for(chunks.get(), timeout=3)
            if piece is None:
                break
            buffer += piece.decode()
            while "\n\n" in buffer:
                block, buffer = buffer.split("\n\n", 1)
                if block.startswith(":"):
                    continue
                payload = json.loads(block.split("data: ", 1)[1])
                frames.append(payload)
                if stop_when(payload):
                    disconnect.set()
                    # 只触发一次。后面心跳或迟到的帧不再重复要求断连。
                    stop_when = _never_stop
        exc = app_task.exception() if app_task.done() else None
        if exc is not None:
            raise exc
        await asyncio.wait_for(app_task, timeout=2)
    finally:
        disconnect.set()
        if not app_task.done():
            app_task.cancel()
            try:
                await app_task
            except asyncio.CancelledError:
                pass
    return frames


def _never_stop(_payload: dict) -> bool:
    return False
