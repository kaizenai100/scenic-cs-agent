"""Gateway 对外抛出的错误。

业务捕获这些类型即可，不要再去解析厂商 SDK 的异常类名。
错误码（code）写进调用记录，排障时和日志对齐。

retryable 只描述「同一条路由上再打一次有没有意义」：
  - 超时、连接失败、429、5xx：上游可能是瞬态的，可以重试；
  - 400/401、缺密钥、调用方取消、熔断打开：再打同一下不会变好。
换一条备路由是另一件事，由 Gateway 的 fallback 决定，不靠这个标志。
"""

from __future__ import annotations

from scenic_cs_agent.runtime.model_types import ModelResult


class GatewayError(Exception):
    """所有网关错误的基类。code 稳定，文案可以改。"""

    def __init__(self, message: str, *, code: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class UpstreamError(GatewayError):
    """上游调用失败。

    适配器和测试都可以主动抛它，用来表达「这是 429 / 5xx / 超时」而不必依赖 openai 的异常类。
    真实 SDK 异常不必改成这个类型：Gateway 会在边界上把它们归类成同样的 code / retryable。
    """


class RateLimitExceeded(GatewayError):
    """进程内限流拒绝。还没打到上游。"""

    def __init__(self, retry_after_seconds: float | None = None) -> None:
        self.retry_after_seconds = retry_after_seconds
        if retry_after_seconds is None:
            message = "进程内限流已触发"
        else:
            message = f"进程内限流已触发，大约 {retry_after_seconds:.2f}s 后可再试"
        super().__init__(message, code="rate_limited", retryable=False)


class CircuitOpenError(GatewayError):
    """该 provider 的熔断是打开的，本次不打上游。"""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(
            f"熔断打开，拒绝调用 provider={provider}",
            code="circuit_open",
            retryable=False,
        )


class StreamCancelled(GatewayError):
    """调用方取消了流式请求。不是上游故障，不能当成失败去推熔断计数。"""

    def __init__(self) -> None:
        super().__init__("流式调用已取消", code="cancelled", retryable=False)


class PartialStreamError(GatewayError):
    """已经向调用方交出过事件，不能重试，也不能换路由把后半段接上。

    换路由续写会把两段互不相关的输出拼成一条回复。调用方只能把这次当作失败，
    自己决定要不要用新的请求重来。
    """

    def __init__(self, provider: str, cause: GatewayError) -> None:
        self.provider = provider
        self.cause = cause
        super().__init__(
            f"流式已向调用方输出，不能换路由续写（provider={provider}，原因={cause.code}）",
            code="stream_partial",
            retryable=False,
        )


class AllRoutesFailed(GatewayError):
    """主路由和备路由都没能完成本次调用。code 取最后一条失败，方便记录。"""

    def __init__(self, errors: list[GatewayError]) -> None:
        self.errors = list(errors)
        last = self.errors[-1] if self.errors else None
        code = last.code if last is not None else "no_route"
        detail = ", ".join(f"{err.code}" for err in self.errors) or "no_route"
        super().__init__(f"所有路由均失败：{detail}", code=code, retryable=False)


class StructuredOutputError(GatewayError):
    """校验失败，并且修复重试已经用完。

    result 留着，调用方可以自己决定兜底话术。
    Gateway 不在这里编业务回复——那是步骤 5 结构化执行器的职责，这里不拆那一层。
    """

    def __init__(self, detail: str, result: ModelResult) -> None:
        self.result = result
        super().__init__(detail, code="structured_invalid", retryable=False)
