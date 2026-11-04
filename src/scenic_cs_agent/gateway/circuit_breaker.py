"""按 provider 的熔断器。

为什么按 provider 而不是按别名：default 和 fast 如果打的是同一个上游，
一个别名已经连续失败，另一个别名不应该继续把流量打过去。
备路由是另一个 provider 时不受影响，fallback 才有意义。

状态：
  closed    正常放行，失败次数累加，到阈值就打开；
  open      冷却时间内直接拒绝，不打上游；
  half_open 冷却结束，放进少量探测；成功则关上，失败则重新打开并重新计冷却。

取消、缺密钥不算失败。那些不是上游故障，算进去会把熔断误打开。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from enum import Enum

from scenic_cs_agent.gateway.clock import Clock
from scenic_cs_agent.gateway.errors import CircuitOpenError, GatewayError

logger = logging.getLogger("scenic_cs_agent.gateway")


class _State(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """一个 provider 一把锁。Gateway 上的并发调用不能把失败次数写丢。"""

    def __init__(
        self,
        provider: str,
        *,
        threshold: int,
        cooldown_seconds: float,
        half_open_max_probes: int,
        clock: Clock,
    ) -> None:
        if threshold < 1:
            raise ValueError("failure_threshold 必须 >= 1")
        if cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds 必须 > 0")
        if half_open_max_probes < 1:
            raise ValueError("half_open_max_probes 必须 >= 1")
        self.provider = provider
        self._threshold = threshold
        self._cooldown = cooldown_seconds
        self._half_open_max = half_open_max_probes
        self._clock = clock
        self._state = _State.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        # 半开时正在飞的探测数。不限制的话，冷却一到会把积压请求全放出去，熔断就白做了。
        self._half_open_inflight = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        """给观测和测试看。冷却已经结束时，读状态会切到半开，但还没有放行。"""
        with self._lock:
            self._maybe_half_open_locked()
            return self._state.value

    def before_call(self) -> None:
        """允许则返回；半开时占用一个探测名额。拒绝抛 CircuitOpenError。"""
        with self._lock:
            self._maybe_half_open_locked()
            if self._state is _State.OPEN:
                raise CircuitOpenError(self.provider)
            if self._state is _State.HALF_OPEN:
                if self._half_open_inflight >= self._half_open_max:
                    raise CircuitOpenError(self.provider)
                self._half_open_inflight += 1

    def record_success(self) -> None:
        """探测或普通调用成功：计数清零，熔断关上。"""
        with self._lock:
            self._failures = 0
            self._half_open_inflight = 0
            self._state = _State.CLOSED

    def record_failure(self) -> None:
        """上游失败。半开失败立刻重新打开；关闭态累加到阈值才打开。"""
        with self._lock:
            if self._state is _State.HALF_OPEN:
                self._trip_locked()
                return
            self._failures += 1
            if self._failures >= self._threshold:
                self._trip_locked()

    def record_neutral(self) -> None:
        """取消或缺密钥：把半开探测名额还回去，失败计数不动。"""
        with self._lock:
            if self._half_open_inflight > 0:
                self._half_open_inflight -= 1

    def _maybe_half_open_locked(self) -> None:
        if self._state is _State.OPEN and (self._clock.monotonic() - self._opened_at) >= self._cooldown:
            self._state = _State.HALF_OPEN
            self._half_open_inflight = 0

    def _trip_locked(self) -> None:
        self._state = _State.OPEN
        self._opened_at = self._clock.monotonic()
        self._half_open_inflight = 0
        logger.warning(
            "熔断打开 provider=%s threshold=%s cooldown=%ss",
            self.provider,
            self._threshold,
            self._cooldown,
        )


class CircuitBreakerGroup:
    """一个 Gateway 实例里，每个 provider 一把熔断器，别名之间共享。"""

    def __init__(
        self,
        providers: Iterable[str],
        *,
        threshold: int,
        cooldown_seconds: float,
        half_open_max_probes: int,
        clock: Clock,
    ) -> None:
        self._breakers = {
            name: CircuitBreaker(
                name,
                threshold=threshold,
                cooldown_seconds=cooldown_seconds,
                half_open_max_probes=half_open_max_probes,
                clock=clock,
            )
            for name in providers
        }

    def states(self) -> dict[str, str]:
        """每个 provider 当前的熔断状态：closed / open / half_open。

        读状态时，冷却已经结束的会先切到半开。这是给就绪检查看的，不是放行。
        """

        return {name: breaker.state for name, breaker in self._breakers.items()}

    def for_provider(self, name: str) -> CircuitBreaker:
        try:
            return self._breakers[name]
        except KeyError as exc:
            raise GatewayError(
                f"没有熔断器的 provider：{name}",
                code="unknown_provider",
                retryable=False,
            ) from exc
