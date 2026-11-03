"""进程内 LLM Gateway。

业务、Agent Loop、示例只通过这里调模型。密钥、选哪家上游、重试、熔断、限流、调用记录
都留在这一层。适配器仍然只负责「把 ModelRequest 翻译成某一家的 SDK 调用」。

这不是一个要单独部署的 HTTP 服务。它和调用方在同一个进程里，
complete / stream 就是入口。Agent Loop 继续收 ModelAdapter：本类实现了 generate，
所以可以把 Gateway 直接传进 run_agent_loop，步骤 7 的脚本适配器不用改签名。

流式取消是协作式的：Gateway 在每条事件之间看 cancel_event，并 close 上游迭代器。
适配器若在 chunk 循环里也看 cancel_event，才能把已经打开的 HTTP 流关掉。
一次阻塞中的建连无法从另一个线程打断，这是当前骨架的边界。
"""

from __future__ import annotations

import inspect
import logging
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field

from scenic_cs_agent.gateway.circuit_breaker import CircuitBreaker, CircuitBreakerGroup
from scenic_cs_agent.gateway.classify import NEUTRAL_CODES, classify_exception
from scenic_cs_agent.gateway.clock import Clock, SystemClock
from scenic_cs_agent.gateway.errors import (
    AllRoutesFailed,
    CircuitOpenError,
    GatewayError,
    PartialStreamError,
    RateLimitExceeded,
    StreamCancelled,
    StructuredOutputError,
)
from scenic_cs_agent.gateway.observe import AttemptRecord, CallRecord, CallStatus, RecordSink
from scenic_cs_agent.gateway.rate_limit import TokenBucket
from scenic_cs_agent.gateway.retry import RetryPolicy
from scenic_cs_agent.gateway.routing import AliasSpec, ProviderBinding, ResolvedRoute, Router
from scenic_cs_agent.gateway.structured import with_repair_feedback
from scenic_cs_agent.runtime.model_types import (
    ModelAdapter,
    ModelCapabilities,
    ModelRequest,
    ModelResult,
    ModelStreamEvent,
)

logger = logging.getLogger("scenic_cs_agent.gateway")

# 校验函数：返回 None 表示通过，返回错误文本表示要带回去让模型改。
OutputValidator = Callable[[ModelResult], str | None]


class _RouteFailed(Exception):
    """只在 Gateway 内部传递「这条路由不行，试下一条」。不要让它漏到业务。"""

    def __init__(self, error: GatewayError) -> None:
        self.error = error
        super().__init__(error.code)


@dataclass
class _Stats:
    """一次用户调用的可变记账。结束时收成不可变的 CallRecord。"""

    call_id: str
    alias: str
    stream: bool
    status: CallStatus = "error"
    provider: str | None = None
    model: str | None = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    attempts: int = 0
    fallback: bool = False
    error_code: str | None = None
    repair_count: int = 0
    retries: int = 0
    attempt_log: list[AttemptRecord] = field(default_factory=list)

    def to_record(self) -> CallRecord:
        return CallRecord(
            call_id=self.call_id,
            alias=self.alias,
            provider=self.provider,
            model=self.model,
            stream=self.stream,
            status=self.status,
            latency_ms=self.latency_ms,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            attempts=self.attempts,
            retries=self.retries,
            fallback=self.fallback,
            error_code=self.error_code,
            repair_count=self.repair_count,
            attempt_log=tuple(self.attempt_log),
        )


class LLMGateway(ModelAdapter):
    """统一入口。生产路径用 complete / stream；generate 是给现有 Agent Loop 的适配器兼容口。"""

    name = "llm-gateway"
    # 真实能力取决于被选中的适配器。这里给一个保守声明，避免有人把 Gateway 当「什么都能做」。
    capabilities = ModelCapabilities(
        chat_completions=True,
        responses=False,
        structured_output="json_mode",
        tool_calling=True,
        supports_temperature=True,
        supports_top_p=True,
    )

    def __init__(
        self,
        *,
        bindings: Mapping[str, ProviderBinding],
        aliases: Mapping[str, AliasSpec],
        default_alias: str,
        retry: RetryPolicy | None = None,
        clock: Clock | None = None,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
        half_open_max_probes: int = 1,
        rate_limit_enabled: bool = True,
        requests_per_minute: int = 60,
        burst: int = 10,
        structured_output_retries: int = 1,
        record_capacity: int = 500,
        on_record: Callable[[CallRecord], None] | None = None,
    ) -> None:
        if structured_output_retries < 0:
            raise ValueError("structured_output_retries 不能为负")
        self._clock = clock or SystemClock()
        self._retry = retry or RetryPolicy()
        self._router = Router(
            bindings=dict(bindings),
            aliases=dict(aliases),
            default_alias=default_alias,
        )
        self._circuits = CircuitBreakerGroup(
            bindings.keys(),
            threshold=failure_threshold,
            cooldown_seconds=cooldown_seconds,
            half_open_max_probes=half_open_max_probes,
            clock=self._clock,
        )
        self._rate = TokenBucket(
            requests_per_minute=requests_per_minute,
            burst=burst,
            clock=self._clock,
            enabled=rate_limit_enabled,
        )
        self._structured_retries = structured_output_retries
        self._sink = RecordSink(capacity=record_capacity, on_record=on_record)
        # 用默认路由的能力覆盖类上的保守声明，Agent Loop 若读取 capabilities 能看到真实上游。
        default_routes = self._router.routes_for(default_alias)
        self.capabilities = default_routes[0].adapter.capabilities

    @property
    def records(self) -> tuple[CallRecord, ...]:
        """按时间顺序的调用记录副本。修改返回值不会改到网关内部队列。"""
        return self._sink.snapshot()

    def breaker_state(self, provider: str) -> str:
        """closed / open / half_open。给示例和测试观察熔断，不作为业务分支的依据。"""
        return self._circuits.for_provider(provider).state

    def generate(self, request: ModelRequest) -> ModelResult:
        """Agent Loop 的入口。别名固定走 default_alias。

        要指定 fast 或其他别名，请直接调 complete(alias=...)。
        Loop 的函数签名仍然只认 ModelAdapter，步骤 7 不必改。
        """
        return self.complete(request)

    def complete(
        self,
        request: ModelRequest,
        *,
        alias: str | None = None,
        output_validator: OutputValidator | None = None,
        max_repairs: int | None = None,
    ) -> ModelResult:
        """非流式。一次业务调用：限流一次，然后按路由重试 / fallback，可选做结构化修复。"""
        stats = _Stats(
            call_id=_new_call_id(),
            alias=alias or self._router.default_alias,
            stream=False,
        )
        started = self._clock.monotonic()
        try:
            self._acquire_rate_limit(stats)
            result = self._complete_with_repairs(
                request,
                alias=alias,
                output_validator=output_validator,
                max_repairs=max_repairs,
                stats=stats,
            )
        except CircuitOpenError as exc:
            self._mark_terminal(stats, "circuit_open", exc)
            raise
        except GatewayError as exc:
            self._mark_terminal(stats, "error", exc)
            raise
        except Exception as exc:
            classified = classify_exception(exc, self._retry.retryable_codes)
            self._mark_terminal(stats, "error", classified)
            raise classified from exc
        else:
            stats.status = "ok"
            stats.error_code = None
            return result
        finally:
            stats.latency_ms = max(0.0, self._clock.monotonic() - started) * 1000.0
            self._sink.append(stats.to_record())

    def stream(
        self,
        request: ModelRequest,
        *,
        alias: str | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[ModelStreamEvent]:
        """流式。事件形状与适配器一致：text.delta / model.finished / model.usage。

        调用方还没收到任何事件时，失败可以重试或换路由。
        一旦已经 yield 出去，就不能再换：调用方手里已经有前半段，拼上另一家的后半段是错的。
        """
        stats = _Stats(
            call_id=_new_call_id(),
            alias=alias or self._router.default_alias,
            stream=True,
        )
        started = self._clock.monotonic()
        try:
            self._acquire_rate_limit(stats)
            try:
                yield from self._stream_routes(request, alias=alias, cancel_event=cancel_event, stats=stats)
            except StreamCancelled as exc:
                self._mark_terminal(stats, "cancelled", exc)
                raise
            except GeneratorExit:
                # 调用方提前停掉迭代，视作取消。必须原样再抛，否则生成器会被当成正常结束。
                stats.status = "cancelled"
                stats.error_code = "cancelled"
                raise
            except CircuitOpenError as exc:
                self._mark_terminal(stats, "circuit_open", exc)
                raise
            except GatewayError as exc:
                self._mark_terminal(stats, "error", exc)
                raise
            except Exception as exc:
                classified = classify_exception(exc, self._retry.retryable_codes)
                self._mark_terminal(stats, "error", classified)
                raise classified from exc
            else:
                stats.status = "ok"
                stats.error_code = None
        finally:
            stats.latency_ms = max(0.0, self._clock.monotonic() - started) * 1000.0
            self._sink.append(stats.to_record())

    def _acquire_rate_limit(self, stats: _Stats) -> None:
        try:
            self._rate.acquire()
        except RateLimitExceeded as exc:
            stats.status = "rate_limited"
            stats.error_code = exc.code
            raise

    def _mark_terminal(self, stats: _Stats, status: CallStatus, exc: GatewayError) -> None:
        # 限流已经在 _acquire_rate_limit 里写过状态，不要被外层 except GatewayError 盖掉。
        if stats.status == "rate_limited":
            return
        stats.status = status
        stats.error_code = exc.code
        if isinstance(exc, CircuitOpenError) and stats.provider is None:
            stats.provider = exc.provider
        if isinstance(exc, PartialStreamError) and stats.provider is None:
            stats.provider = exc.provider

    def _complete_with_repairs(
        self,
        request: ModelRequest,
        *,
        alias: str | None,
        output_validator: OutputValidator | None,
        max_repairs: int | None,
        stats: _Stats,
    ) -> ModelResult:
        allowed = self._structured_retries if max_repairs is None else max_repairs
        if allowed < 0:
            raise GatewayError("max_repairs 不能为负", code="bad_config", retryable=False)
        # 修复始终挂在最初的请求上，只附最近一次错误，避免提示词越拼越长。
        original = request
        current = request
        while True:
            result = self._complete_transport(current, alias=alias, stats=stats)
            stats.input_tokens += result.input_tokens
            stats.output_tokens += result.output_tokens
            if output_validator is None:
                return result
            error = output_validator(result)
            if error is None:
                return result
            if not isinstance(error, str):
                raise GatewayError(
                    "output_validator 必须返回 str 或 None",
                    code="bad_validator",
                    retryable=False,
                )
            if stats.repair_count >= allowed:
                raise StructuredOutputError(error, result)
            stats.repair_count += 1
            current = with_repair_feedback(original, raw_text=result.text, error=error)

    def _complete_transport(self, request: ModelRequest, *, alias: str | None, stats: _Stats) -> ModelResult:
        routes = self._router.routes_for(alias)
        errors: list[GatewayError] = []
        for index, route in enumerate(routes):
            if index > 0:
                stats.fallback = True
                logger.info(
                    "gateway fallback alias=%s to provider=%s model=%s",
                    stats.alias,
                    route.provider,
                    route.model,
                )
            try:
                return self._retry_route(route, request, stats)
            except _RouteFailed as failed:
                errors.append(failed.error)
                self._remember_route(stats, route)
        if errors and all(isinstance(err, CircuitOpenError) for err in errors):
            # 每条路由都是熔断拒绝：对调用方来说这就是「熔断打开」，不是一次普通上游失败。
            raise errors[-1]
        raise AllRoutesFailed(errors)

    def _retry_route(self, route: ResolvedRoute, request: ModelRequest, stats: _Stats) -> ModelResult:
        last: GatewayError | None = None
        for attempt in range(1, self._retry.max_attempts_per_route + 1):
            try:
                result = self._one_generate(route, request)
            except CircuitOpenError as exc:
                stats.attempt_log.append(
                    AttemptRecord(
                        provider=route.provider,
                        model=route.model,
                        attempt=0,
                        error_code=exc.code,
                        skipped=True,
                    )
                )
                raise _RouteFailed(exc) from exc
            except Exception as exc:
                classified = self._as_gateway_error(exc)
                stats.attempts += 1
                stats.attempt_log.append(
                    AttemptRecord(
                        provider=route.provider,
                        model=route.model,
                        attempt=attempt,
                        error_code=classified.code,
                    )
                )
                last = classified
                # 熔断刚刚被这次失败打开时，同一路由上的下一次只会吃到 CircuitOpenError。
                opened = self._circuits.for_provider(route.provider).state == "open"
                give_up = (
                    classified.code in NEUTRAL_CODES
                    or not classified.retryable
                    or attempt >= self._retry.max_attempts_per_route
                    or opened
                )
                if give_up:
                    raise _RouteFailed(classified) from exc
                stats.retries += 1
                self._clock.sleep(self._retry.delay_after(attempt))
                continue
            else:
                stats.attempts += 1
                stats.provider = route.provider
                stats.model = route.model
                stats.attempt_log.append(
                    AttemptRecord(
                        provider=route.provider,
                        model=route.model,
                        attempt=attempt,
                        error_code=None,
                    )
                )
                return result
        raise _RouteFailed(last or GatewayError("路由重试耗尽", code="retries_exhausted", retryable=False))

    def _one_generate(self, route: ResolvedRoute, request: ModelRequest) -> ModelResult:
        breaker = self._circuits.for_provider(route.provider)
        breaker.before_call()
        try:
            result = route.adapter.generate(request)
        except Exception as exc:
            classified = self._note_upstream_failure(breaker, exc)
            if classified is exc:
                raise
            raise classified from exc
        breaker.record_success()
        return result

    def _stream_routes(
        self,
        request: ModelRequest,
        *,
        alias: str | None,
        cancel_event: threading.Event | None,
        stats: _Stats,
    ) -> Iterator[ModelStreamEvent]:
        routes = self._router.routes_for(alias)
        errors: list[GatewayError] = []
        for index, route in enumerate(routes):
            if index > 0:
                stats.fallback = True
                logger.info(
                    "gateway stream fallback alias=%s to provider=%s model=%s",
                    stats.alias,
                    route.provider,
                    route.model,
                )
            try:
                yield from self._retry_stream(route, request, cancel_event, stats)
                return
            except _RouteFailed as failed:
                errors.append(failed.error)
                self._remember_route(stats, route)
        if errors and all(isinstance(err, CircuitOpenError) for err in errors):
            raise errors[-1]
        raise AllRoutesFailed(errors)

    def _retry_stream(
        self,
        route: ResolvedRoute,
        request: ModelRequest,
        cancel_event: threading.Event | None,
        stats: _Stats,
    ) -> Iterator[ModelStreamEvent]:
        last: GatewayError | None = None
        for attempt in range(1, self._retry.max_attempts_per_route + 1):
            if cancel_event is not None and cancel_event.is_set():
                raise StreamCancelled()
            breaker = self._circuits.for_provider(route.provider)
            try:
                breaker.before_call()
            except CircuitOpenError as exc:
                stats.attempt_log.append(
                    AttemptRecord(
                        provider=route.provider,
                        model=route.model,
                        attempt=0,
                        error_code=exc.code,
                        skipped=True,
                    )
                )
                raise _RouteFailed(exc) from exc

            delivered = False
            try:
                iterator = _open_stream(route.adapter, request, cancel_event)
            except Exception as exc:
                classified = self._note_upstream_failure(breaker, exc)
                last = self._record_stream_failure(stats, route, attempt, classified)
                if self._should_give_up_stream(route, classified, attempt):
                    raise _RouteFailed(classified) from exc
                stats.retries += 1
                self._clock.sleep(self._retry.delay_after(attempt))
                continue

            try:
                for event in iterator:
                    if cancel_event is not None and cancel_event.is_set():
                        raise StreamCancelled()
                    delivered = True
                    if event.type == "model.usage":
                        stats.input_tokens = int(event.data.get("input_tokens") or 0)
                        stats.output_tokens = int(event.data.get("output_tokens") or 0)
                    yield event
                if cancel_event is not None and cancel_event.is_set():
                    # 适配器看到取消后直接 return，生成器正常结束。这里仍要告诉调用方是取消，不是成功。
                    raise StreamCancelled()
            except StreamCancelled:
                # 流已经打开：算一次上游尝试，并记下是哪条路由，取消也要能对上 provider。
                self._record_stream_cancel(stats, route, attempt)
                breaker.record_neutral()
                raise
            except GeneratorExit:
                # 调用方直接关掉生成器，语义同样是取消。必须继续抛，不能当成正常结束。
                self._record_stream_cancel(stats, route, attempt)
                breaker.record_neutral()
                raise
            except Exception as exc:
                classified = self._note_upstream_failure(breaker, exc)
                last = self._record_stream_failure(stats, route, attempt, classified)
                if delivered:
                    # 事件已经交给调用方。再重试或换路由都会把另一段输出接在后面。
                    raise PartialStreamError(route.provider, classified) from exc
                if self._should_give_up_stream(route, classified, attempt):
                    raise _RouteFailed(classified) from exc
                stats.retries += 1
                self._clock.sleep(self._retry.delay_after(attempt))
                continue
            else:
                breaker.record_success()
                stats.attempts += 1
                stats.provider = route.provider
                stats.model = route.model
                stats.attempt_log.append(
                    AttemptRecord(
                        provider=route.provider,
                        model=route.model,
                        attempt=attempt,
                        error_code=None,
                    )
                )
                return
            finally:
                _close_iterator(iterator)
        raise _RouteFailed(last or GatewayError("流式重试耗尽", code="retries_exhausted", retryable=False))

    def _record_stream_cancel(self, stats: _Stats, route: ResolvedRoute, attempt: int) -> None:
        stats.attempts += 1
        stats.provider = route.provider
        stats.model = route.model
        stats.attempt_log.append(
            AttemptRecord(
                provider=route.provider,
                model=route.model,
                attempt=attempt,
                error_code="cancelled",
            )
        )

    def _record_stream_failure(
        self,
        stats: _Stats,
        route: ResolvedRoute,
        attempt: int,
        classified: GatewayError,
    ) -> GatewayError:
        stats.attempts += 1
        stats.attempt_log.append(
            AttemptRecord(
                provider=route.provider,
                model=route.model,
                attempt=attempt,
                error_code=classified.code,
            )
        )
        return classified

    def _should_give_up_stream(self, route: ResolvedRoute, classified: GatewayError, attempt: int) -> bool:
        opened = self._circuits.for_provider(route.provider).state == "open"
        return (
            classified.code in NEUTRAL_CODES
            or not classified.retryable
            or attempt >= self._retry.max_attempts_per_route
            or opened
        )

    def _note_upstream_failure(self, breaker: CircuitBreaker, exc: Exception) -> GatewayError:
        """记熔断，并把异常收成 GatewayError。取消和缺密钥不计失败。"""
        if isinstance(exc, StreamCancelled):
            breaker.record_neutral()
            return exc
        classified = exc if isinstance(exc, GatewayError) else classify_exception(exc, self._retry.retryable_codes)
        if classified.code in NEUTRAL_CODES:
            breaker.record_neutral()
        else:
            breaker.record_failure()
        return classified

    def _as_gateway_error(self, exc: Exception) -> GatewayError:
        if isinstance(exc, GatewayError):
            return exc
        return classify_exception(exc, self._retry.retryable_codes)

    @staticmethod
    def _remember_route(stats: _Stats, route: ResolvedRoute) -> None:
        # 全部失败时，记录停在最后一条试过的路由上，和 error_code 对齐。
        stats.provider = route.provider
        stats.model = route.model


def _new_call_id() -> str:
    return uuid.uuid4().hex[:16]


def _open_stream(
    adapter: ModelAdapter,
    request: ModelRequest,
    cancel_event: threading.Event | None,
) -> Iterator[ModelStreamEvent]:
    """调用适配器的 stream。只有签名里声明了 cancel_event 才传入，旧适配器不用改。"""
    stream = getattr(adapter, "stream", None)
    if not callable(stream):
        raise GatewayError(
            f"适配器 {getattr(adapter, 'name', type(adapter).__name__)} 没有 stream()",
            code="stream_unsupported",
            retryable=False,
        )
    try:
        parameters = inspect.signature(stream).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "cancel_event" in parameters:
        return stream(request, cancel_event=cancel_event)
    return stream(request)


def _close_iterator(iterator: object) -> None:
    """关掉生成器，让适配器的 finally 有机会关闭 HTTP 流。关失败不能掩盖原来的错误。"""
    close = getattr(iterator, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:
        logger.debug("关闭上游流失败", exc_info=True)
