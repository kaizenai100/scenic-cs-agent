"""没有密钥时的占位适配器。

公开仓的 yaml 只写 ${VAR}。测试和示例在没配密钥时也要能把 Gateway 建出来，
但不能因此去 new 一个真实客户端——那会在导入或第一次调用时就去读环境变量，甚至发出网络请求。

真调用到这里说明：路由选中了这个 provider，可是密钥仍是空的。
立刻用稳定错误码失败，并且不要重试（再试还是没有密钥）。
"""

from __future__ import annotations

from collections.abc import Iterator

from scenic_cs_agent.gateway.errors import UpstreamError
from scenic_cs_agent.runtime.model_types import (
    ModelAdapter,
    ModelCapabilities,
    ModelRequest,
    ModelResult,
    ModelStreamEvent,
)


class UnconfiguredAdapter(ModelAdapter):
    name = "unconfigured"
    capabilities = ModelCapabilities(
        chat_completions=False,
        responses=False,
        structured_output="prompt_only",
        tool_calling=False,
        supports_temperature=False,
        supports_top_p=False,
    )

    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.name = f"unconfigured:{provider}"

    def generate(self, request: ModelRequest) -> ModelResult:
        del request
        raise UpstreamError(
            f"provider={self.provider} 没有配置 API Key，拒绝调用上游",
            code="missing_api_key",
            retryable=False,
        )

    def stream(
        self,
        request: ModelRequest,
        *,
        cancel_event: object | None = None,
    ) -> Iterator[ModelStreamEvent]:
        del request, cancel_event
        raise UpstreamError(
            f"provider={self.provider} 没有配置 API Key，拒绝调用上游",
            code="missing_api_key",
            retryable=False,
        )
