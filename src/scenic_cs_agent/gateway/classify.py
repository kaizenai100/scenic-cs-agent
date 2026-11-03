"""把各式各样的上游异常收成统一的 code + 是否可重试。

适配器继续抛厂商 SDK 的异常即可，不必为了网关改每个 Adapter。
Gateway 只在这一处认识 openai 的异常类型；业务代码看不到它们。
"""

from __future__ import annotations

from scenic_cs_agent.gateway.errors import GatewayError, UpstreamError

# 这些结果不是「上游挂了」：重试没有意义，也不该推动熔断计数。
# cancelled：调用方主动停；missing_api_key：配置问题；stream_unsupported：适配器能力不足。
NEUTRAL_CODES = frozenset({"missing_api_key", "cancelled", "stream_unsupported", "unknown_alias"})


def classify_exception(exc: Exception, retryable_codes: frozenset[str]) -> GatewayError:
    """已是 GatewayError 的原样返回（尊重显式 retryable）；其余按类型和状态码归类。"""
    if isinstance(exc, GatewayError):
        return exc
    code = _code_of(exc)
    return UpstreamError(str(exc), code=code, retryable=code in retryable_codes)


def _code_of(exc: Exception) -> str:
    """抽出稳定错误码。认不出的一律 upstream_error，默认不重试，避免把 bug 当成瞬态故障。"""
    openai = _openai()
    if openai is not None:
        # 超时 / 连接失败没有 HTTP 状态码，单独命名，配置里用 timeout、connection 引用。
        if isinstance(exc, openai.APITimeoutError):
            return "timeout"
        if isinstance(exc, openai.APIConnectionError):
            return "connection"
        if isinstance(exc, openai.RateLimitError):
            return "429"
        status_error = getattr(openai, "APIStatusError", None)
        if status_error is not None and isinstance(exc, status_error):
            return str(exc.status_code)

    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ConnectionError):
        return "connection"

    # 鸭子类型：有些包装异常只带 status_code 或 code，不必强求是 openai 的类。
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return str(status)
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code
    return "upstream_error"


def _openai() -> object | None:
    """openai 是项目依赖；导入失败时仍能给 Fake Adapter 用，不让归类器把进程拖垮。"""
    try:
        import openai
    except ImportError:  # pragma: no cover - 运行环境装了依赖时走不到
        return None
    return openai
