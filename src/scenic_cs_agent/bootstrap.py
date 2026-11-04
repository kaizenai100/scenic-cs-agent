"""进程启动时按顺序装配，关闭时反过来释放。

顺序：
  配置（不合法就立刻失败）→ 日志 → Langfuse（配置开启并且密钥齐全才连）
  → Prompt 注册表 → 全进程一个 Gateway → McpHub（required 失败则启动失败）
  → ToolRuntime（本地工具 + MCP 工具）→ RunStore 和 AgentService。

关闭：取消还在跑的 Run → 关闭 MCP → flush Langfuse。
这些对象放在 app.state 上，路由用 Depends 来取。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from scenic_cs_agent.agent.service import AgentService
from scenic_cs_agent.api.routes.health import build_health_router
from scenic_cs_agent.api.routes.runs import DEFAULT_HEARTBEAT_INTERVAL, StartRun, build_runs_router
from scenic_cs_agent.api.run_store import RunState, RunStatus, RunStore
from scenic_cs_agent.api.schemas import CreateRunRequest
from scenic_cs_agent.api.streaming import create_stream_app
from scenic_cs_agent.config.loader import AppConfig, load_config
from scenic_cs_agent.gateway.factory import build_gateway_from_config
from scenic_cs_agent.mcp.client import McpHub
from scenic_cs_agent.prompts.registry import PromptRegistry
from scenic_cs_agent.prompts.scenic_presale import render_scenic_presale
from scenic_cs_agent.runtime.model_types import ModelAdapter
from scenic_cs_agent.tools import SCENIC_TOOLS, ToolRuntime

logger = logging.getLogger("scenic_cs_agent.bootstrap")

# 关闭时最多等这么久，让进行中的工具调用自己停下来。
_SHUTDOWN_WAIT_SECONDS = 5.0


def create_app(
    *,
    config: AppConfig | None = None,
    adapters: Mapping[str, ModelAdapter] | None = None,
    source_factory: Callable[..., Any] | None = None,
    heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    store: RunStore | None = None,
    max_pre_text_attempts: int = 2,
) -> FastAPI:
    """生产应用走 lifespan。传入 source_factory 时只装配流式测试路径，不连 MCP。"""

    if source_factory is not None:
        return create_stream_app(
            source_factory=source_factory,
            heartbeat_interval=heartbeat_interval,
            store=store,
            max_pre_text_attempts=max_pre_text_attempts,
        )
    if heartbeat_interval <= 0:
        raise ValueError("heartbeat_interval 必须大于 0")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with assemble(app, config=config, adapters=adapters):
            yield

    app = FastAPI(title="scenic-cs-agent", version="0.1.0", lifespan=lifespan)
    app.state.heartbeat_interval = heartbeat_interval
    if store is not None:
        # 测试可以塞进自己的 Store。lifespan 里若已经有，就不再换。
        app.state.store = store
    app.include_router(build_health_router())
    app.include_router(build_runs_router())
    return app


@asynccontextmanager
async def assemble(
    app: FastAPI,
    *,
    config: AppConfig | None = None,
    adapters: Mapping[str, ModelAdapter] | None = None,
) -> AsyncIterator[None]:
    """真正的启动和关闭。create_app 的 lifespan 只是把它套进 FastAPI。"""

    # 1. 配置。环境名非法、yaml 对不上，这里就失败，不会带着半套依赖往下走。
    app_config = config if config is not None else load_config()
    # 2. 日志。配置已经读完，后面的步骤才能把环境名打出来。
    configure_logging()
    logger.info("加载配置 env=%s", app_config.env)

    langfuse_client: Any | None = None
    hub: McpHub | None = None
    run_store: RunStore | None = getattr(app.state, "store", None)
    try:
        # 3. Langfuse。开关关掉，或密钥不齐，都不建立连接。
        langfuse_client = open_langfuse(app_config)
        # 4. Prompt 注册表。渲染一次，模板坏了就现在失败。
        prompts = PromptRegistry(app_config, langfuse_client)
        warmed = render_scenic_presale(registry=prompts)
        logger.info(
            "prompt %s label=%s source=%s version=%s",
            warmed.name,
            warmed.label,
            warmed.source,
            warmed.version,
        )
        # 5. 全进程一个 Gateway。业务代码不要再各自 new 客户端。
        gateway = build_gateway_from_config(
            app_config,
            adapters=adapters,
            langfuse_client=langfuse_client,
        )
        # 6. MCP。required 的服务起不来，异常抛出，下面的 yield 不会执行。
        hub = McpHub(app_config.mcp, local_names=[tool.name for tool in SCENIC_TOOLS])
        hub.connect()
        # 7. 本地工具和 MCP 工具放进同一个 Runtime，场景过滤仍由 Runtime 做。
        runtime = ToolRuntime([*SCENIC_TOOLS, *hub.tools])
        # 8. Run 记录和对话服务。
        if run_store is None:
            run_store = RunStore()
        service = AgentService(gateway=gateway, runtime=runtime, prompts=prompts, store=run_store)
        app.state.config = app_config
        app.state.langfuse = langfuse_client
        app.state.prompts = prompts
        app.state.gateway = gateway
        app.state.mcp = hub
        app.state.runtime = runtime
        app.state.store = run_store
        app.state.agent = service
        app.state.start_run = _start_run(service)
        logger.info(
            "进程装配完成 env=%s providers=%s mcp_tools=%d",
            app_config.env,
            ",".join(gateway.breaker_states()) or "（无）",
            len(hub.tools),
        )
        yield
    finally:
        if run_store is not None:
            await cancel_inflight(run_store)
        if hub is not None:
            hub.close()
        flush_langfuse(langfuse_client)


def _start_run(service: AgentService) -> StartRun:
    async def start(state: RunState, body: CreateRunRequest) -> None:
        await service.run(state, message=body.message, system=body.system, scene=body.scene)

    return start


def configure_logging() -> None:
    """没有处理器时补一个。测试已经配过日志时，不把它们换掉。"""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("scenic_cs_agent").setLevel(logging.INFO)


def open_langfuse(config: AppConfig) -> Any | None:
    """配置开启并且公钥、私钥都在，才创建客户端。缺一就不连。"""

    if not config.gateway.observe.langfuse_enabled:
        logger.info("Langfuse 未开启，不建立连接")
        return None
    if not config.langfuse.public_key or not config.langfuse.secret_key:
        logger.info("Langfuse 已开启但缺少密钥，prompt 用本地文件，调用不上报")
        return None
    from langfuse import Langfuse

    logger.info("连接 Langfuse host=%s environment=%s", config.langfuse.host, config.langfuse.environment)
    return Langfuse(
        host=config.langfuse.host,
        public_key=config.langfuse.public_key,
        secret_key=config.langfuse.secret_key,
        environment=config.langfuse.environment,
    )


def flush_langfuse(client: Any | None) -> None:
    """把还留在缓冲区里的观测发出去。失败只记日志，不影响进程退出。"""

    if client is None:
        return
    flush = getattr(client, "flush", None)
    if not callable(flush):
        return
    try:
        flush()
    except Exception:
        logger.warning("Langfuse flush 失败", exc_info=True)


async def cancel_inflight(store: RunStore) -> None:
    """关闭前停掉还在跑的 Run。先协作式取消，再打断正堵在 await 上的任务。"""

    tasks: list[asyncio.Task[None]] = []
    for state in store.all():
        if state.status == RunStatus.RUNNING:
            await store.request_cancel(state)
        task = state.task
        if task is not None and not task.done():
            task.cancel()
            tasks.append(task)
    if not tasks:
        return
    _done, pending = await asyncio.wait(tasks, timeout=_SHUTDOWN_WAIT_SECONDS)
    for task in pending:
        logger.warning("关闭时 run 任务未在时限内结束 name=%s", task.get_name())
