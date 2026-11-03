"""进程内 LLM Gateway。

业务代码从这里拿入口，不要直接拿上游密钥或厂商客户端：

    gateway = build_gateway_from_config()
    result = gateway.complete(request)
    for event in gateway.stream(request, cancel_event=cancel):
        ...

适配器仍然按厂商分开。路由、重试、熔断、限流、调用记录在 Gateway。
"""

from scenic_cs_agent.gateway.errors import (
    AllRoutesFailed,
    CircuitOpenError,
    GatewayError,
    PartialStreamError,
    RateLimitExceeded,
    StreamCancelled,
    StructuredOutputError,
    UpstreamError,
)
from scenic_cs_agent.gateway.factory import build_gateway_from_config
from scenic_cs_agent.gateway.gateway import LLMGateway
from scenic_cs_agent.gateway.observe import AttemptRecord, CallRecord
from scenic_cs_agent.gateway.retry import RetryPolicy
from scenic_cs_agent.gateway.routing import AliasSpec, ProviderBinding, RouteSpec

__all__ = [
    "AliasSpec",
    "AllRoutesFailed",
    "AttemptRecord",
    "CallRecord",
    "CircuitOpenError",
    "GatewayError",
    "LLMGateway",
    "PartialStreamError",
    "ProviderBinding",
    "RateLimitExceeded",
    "RetryPolicy",
    "RouteSpec",
    "StreamCancelled",
    "StructuredOutputError",
    "UpstreamError",
    "build_gateway_from_config",
]
