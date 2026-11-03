"""同一条路由上的有限次退避。

只解决瞬态失败（超时、429、5xx）。次数和等待都封顶，避免一次用户请求把上游打成重试风暴。
指数退避但不加抖动：骨架要让测试可重复；真要防惊群，以后可以在这里乘一个随机系数，策略对象不用换。
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _default_retryable_codes() -> frozenset[str]:
    # 408/409 不放进默认集合：需求只要求超时、429、5xx。需要的话在 yaml 里加。
    return frozenset({"timeout", "connection", "429", "500", "502", "503", "504"})


@dataclass(frozen=True)
class RetryPolicy:
    """一条路由内部的重试预算。max_attempts_per_route 含第一次，所以 2 表示失败后再试 1 次。"""

    max_attempts_per_route: int = 2
    base_delay_seconds: float = 0.2
    max_delay_seconds: float = 2.0
    retryable_codes: frozenset[str] = field(default_factory=_default_retryable_codes)

    def __post_init__(self) -> None:
        if self.max_attempts_per_route < 1:
            raise ValueError("max_attempts_per_route 必须 >= 1")
        if self.base_delay_seconds < 0 or self.max_delay_seconds < 0:
            raise ValueError("重试等待不能为负")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds 不能小于 base_delay_seconds")
        # dataclass frozen 时不能直接改字段，用 object.__setattr__ 把 list 收成集合，调用方传 list 也能用。
        codes = self.retryable_codes
        if not isinstance(codes, frozenset):
            object.__setattr__(self, "retryable_codes", frozenset(codes))

    def delay_after(self, failed_attempt: int) -> float:
        """第 failed_attempt 次失败之后等待多久。第 1 次等 base，之后翻倍，不超过 max。"""
        if failed_attempt < 1 or self.base_delay_seconds == 0:
            return 0.0
        raw = self.base_delay_seconds * (2 ** (failed_attempt - 1))
        if raw > self.max_delay_seconds:
            return self.max_delay_seconds
        return raw
