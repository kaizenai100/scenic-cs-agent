"""测试注入用的流式生产者装配。

生产对话不走这里，走 AgentService。保留这条路径，是为了继续覆盖
断连不取消、Last-Event-ID 续传、显式取消，以及正文前后的失败处理。
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI

from scenic_cs_agent.api.producer import ModelStreamSource, run_producer
from scenic_cs_agent.api.routes.runs import DEFAULT_HEARTBEAT_INTERVAL, build_runs_router
from scenic_cs_agent.api.run_store import RunState, RunStore
from scenic_cs_agent.api.schemas import CreateRunRequest
from scenic_cs_agent.runtime.model_types import ModelRequest

# 测试没传 system 时用的短边界。不编造票价。
_STREAM_SYSTEM = (
    "你是景区门票售前助手。"
    "不能编造票价或开放时间；没查到资料就说明需要先查，并追问景区名和日期。"
)


def create_stream_app(
    *,
    source_factory: Callable[[], ModelStreamSource],
    heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    store: RunStore | None = None,
    max_pre_text_attempts: int = 2,
) -> FastAPI:
    """只挂上 Run 路由，不连 MCP，也不建 Gateway。"""

    if heartbeat_interval <= 0:
        raise ValueError("heartbeat_interval 必须大于 0")
    app = FastAPI(title="scenic-cs-agent", version="0.1.0")
    run_store = store or RunStore()
    app.state.store = run_store
    app.state.heartbeat_interval = heartbeat_interval

    async def start_run(state: RunState, body: CreateRunRequest) -> None:
        model_request = ModelRequest(
            system=body.system or _STREAM_SYSTEM,
            user=body.message,
            max_output_tokens=512,
        )
        # 每个 Run 一份新的 source。脚本模型把列表迭代一次就耗尽了，不能跨 Run 复用。
        await run_producer(
            run_store,
            state,
            source_factory(),
            model_request,
            max_pre_text_attempts=max_pre_text_attempts,
        )

    app.state.start_run = start_run
    app.include_router(build_runs_router())
    return app
