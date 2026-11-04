"""存活和就绪。

/healthz 只回答进程还在不在。
/readyz 才看 Gateway 熔断和 MCP 连接，调用方自己决定要不要把流量切走。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from scenic_cs_agent.api.deps import get_gateway, get_mcp
from scenic_cs_agent.api.schemas import HealthResponse, McpServerView, ReadyzResponse
from scenic_cs_agent.gateway.gateway import LLMGateway
from scenic_cs_agent.mcp.client import McpHub


def build_health_router() -> APIRouter:
    """每个应用一份，避免同一个路由器挂进两个进程。"""

    router = APIRouter()
    router.add_api_route("/healthz", healthz, methods=["GET"], response_model=HealthResponse)
    router.add_api_route("/readyz", readyz, methods=["GET"], response_model=ReadyzResponse)
    return router


async def healthz() -> HealthResponse:
    """进程还活着就 200。这里不连模型，也不 ping MCP。"""

    return HealthResponse(status="ok")


async def readyz(
    gateway: LLMGateway = Depends(get_gateway),
    hub: McpHub = Depends(get_mcp),
) -> ReadyzResponse:
    """返回每个 provider 的熔断状态，以及每台 MCP 的连接状态和错误。

    必需的 MCP 没连上时进程起不来，所以这里看不到那种失败。
    熔断打开或非必需 MCP 报错时 status 为 degraded，HTTP 仍是 200：
    进程还能用本地工具回答，要不要摘流量由调用方决定。
    """

    states = gateway.breaker_states()
    errors = list(hub.errors)
    mcp = {
        name: McpServerView(state=state, error=hub.server_errors.get(name, ""))
        for name, state in hub.server_state.items()
    }
    degraded = bool(errors) or any(item.state == "error" for item in mcp.values())
    if any(state != "closed" for state in states.values()):
        degraded = True
    return ReadyzResponse(
        status="degraded" if degraded else "ok",
        gateway=states,
        mcp=mcp,
        errors=errors,
    )
