"""从 app.state 取启动时装配好的对象。

路由不要自己 new Gateway 或 MCP。测试要换实现时，改 app.state，
或者用 app.dependency_overrides 盖住下面这些函数。
"""

from __future__ import annotations

from fastapi import HTTPException, Request

from scenic_cs_agent.api.run_store import RunStore
from scenic_cs_agent.gateway.gateway import LLMGateway
from scenic_cs_agent.mcp.client import McpHub


def _require(request: Request, name: str) -> object:
    value = getattr(request.app.state, name, None)
    if value is None:
        raise HTTPException(status_code=503, detail="服务还没装配完成")
    return value


def get_store(request: Request) -> RunStore:
    """这一进程里的 Run 记录。"""

    store = _require(request, "store")
    if not isinstance(store, RunStore):
        raise HTTPException(status_code=503, detail="服务还没装配完成")
    return store


def get_gateway(request: Request) -> LLMGateway:
    """进程内唯一的 Gateway。"""

    gateway = _require(request, "gateway")
    if not isinstance(gateway, LLMGateway):
        raise HTTPException(status_code=503, detail="服务还没装配完成")
    return gateway


def get_mcp(request: Request) -> McpHub:
    """启动时连上的 MCP。关闭前都是这一个实例。"""

    hub = _require(request, "mcp")
    if not isinstance(hub, McpHub):
        raise HTTPException(status_code=503, detail="服务还没装配完成")
    return hub


def get_heartbeat(request: Request) -> float:
    """SSE 心跳间隔，秒。没配就用路由上的默认值。"""

    value = getattr(request.app.state, "heartbeat_interval", None)
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    raise HTTPException(status_code=503, detail="服务还没装配完成")
