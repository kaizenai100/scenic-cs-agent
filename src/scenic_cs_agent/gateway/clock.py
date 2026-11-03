"""时钟抽象。

重试等待、熔断冷却、令牌桶补桶都要读「现在」和「睡一会」。
测试里如果真的 time.sleep，用例会又慢又容易抖。
所以生产用 SystemClock，测试注入一个只改数字、不阻塞的假时钟。
两类时间用的是同一把钟：冷却跳过 30 秒时，令牌桶也会认为过了 30 秒。
"""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    """单调钟。不要用墙上时间，NTP 回拨会让熔断提前关上、令牌桶多放行。"""

    def monotonic(self) -> float:
        """返回单调递增的秒数。"""

    def sleep(self, seconds: float) -> None:
        """阻塞当前调用 seconds 秒。0 或负数应立刻返回。"""


class SystemClock:
    """进程真实时钟。sleep 会真的让出 CPU。"""

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)
