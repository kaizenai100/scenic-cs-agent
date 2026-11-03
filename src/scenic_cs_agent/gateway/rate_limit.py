"""进程内令牌桶。

防的是「这一个进程」把上游打爆，不是全局限额。
多副本部署时每个进程各有一份额度；骨架阶段不做跨进程存储。

一次业务调用（complete 或 stream）只扣 1 个令牌。
同一调用里的传输重试、fallback、结构化修复不再扣：
那些是网关自己为了完成这一次请求产生的，扣多次会把「用户打了一次」放大成限流。
熔断拒绝也扣 1 个。否则调用方可以靠紧循环，在熔断打开时把 CPU 打满。
"""

from __future__ import annotations

import threading

from scenic_cs_agent.gateway.clock import Clock
from scenic_cs_agent.gateway.errors import RateLimitExceeded


class TokenBucket:
    """容量 = burst，补充速度 = requests_per_minute / 60。冷启动时桶是满的。"""

    def __init__(
        self,
        *,
        requests_per_minute: int,
        burst: int,
        clock: Clock,
        enabled: bool = True,
    ) -> None:
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute 必须 >= 1")
        if burst < 1:
            raise ValueError("burst 必须 >= 1")
        self._enabled = enabled
        self._capacity = float(burst)
        self._tokens = float(burst)
        self._refill_per_sec = requests_per_minute / 60.0
        self._clock = clock
        self._last = clock.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """拿走 1 个令牌。不够就抛 RateLimitExceeded，调用方不应再打上游。"""
        if not self._enabled:
            return
        with self._lock:
            now = self._clock.monotonic()
            elapsed = now - self._last
            if elapsed < 0:
                # 假时钟被往回拨时不倒灌令牌，宁可少放行。
                elapsed = 0.0
            self._last = now
            self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_per_sec)
            if self._tokens < 1.0:
                missing = 1.0 - self._tokens
                retry_after = missing / self._refill_per_sec if self._refill_per_sec else None
                raise RateLimitExceeded(retry_after_seconds=retry_after)
            self._tokens -= 1.0
