"""SSE 帧的纯函数编码。

编码和 FastAPI、事件循环拆开，是为了单测不用起服务就能核对字节。
一条业务事件是一个帧：id / event / data，然后空行。
心跳是注释帧，不带 id，因此不占用 Run 的 seq。
"""

from __future__ import annotations

import json

from scenic_cs_agent.api.events import RunEvent

# 反代（尤其是 nginx）默认会缓冲响应。这三个头告诉它：不要缓存、不要改写、不要攒包。
# media_type 不放在这里，StreamingResponse 用单独的参数设置 text/event-stream。
SSE_RESPONSE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}

SSE_MEDIA_TYPE = "text/event-stream"


def encode_sse(event: RunEvent) -> str:
    """把一条 RunEvent 编成一个 SSE 帧。

    id 用 seq：浏览器重连时会把最后看到的 id 放进 Last-Event-ID。
    event 用业务 type：客户端按类型分发，不必先解析 JSON。
    data 是整份信封的单行 JSON。增量正文里的换行由 JSON 转义，
    不能出现真正的换行，否则会被当成下一个 SSE 字段。
    帧尾的空行是 SSE 规范里的事件分隔符。
    """

    payload = event.model_dump()
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if "\n" in data or "\r" in data:
        raise ValueError("SSE data 必须是单行")
    return f"id: {event.seq}\nevent: {event.type}\ndata: {data}\n\n"


def encode_heartbeat() -> str:
    """心跳注释帧。

    以冒号开头的行是 SSE 注释，EventSource 会丢掉，业务回调不会触发。
    不写 id：心跳不是业务事件，重放和去重都不该看见它，所以不消耗 seq。
    作用只是在模型长时间不吐字时，把中间代理的空闲连接保住（常见超时 30–60 秒）。
    """

    return ": keep-alive\n\n"
