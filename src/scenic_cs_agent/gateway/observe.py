"""一次调用一条记录，外加可选回调。

不落 SQLite：骨架阶段把记录留在内存里，测试和示例可以直接断言。
回调给以后接日志或追踪用。回调抛错只记日志，不能把一次已经成功的模型调用改成失败。
回调里不要再调用 Gateway，否则会重入限流和熔断。
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger("scenic_cs_agent.gateway")

CallStatus = Literal["ok", "error", "cancelled", "rate_limited", "circuit_open"]


@dataclass(frozen=True)
class AttemptRecord:
    """路由上的一次尝试。skipped=True 表示熔断直接拒绝，没有打到适配器。"""

    provider: str
    model: str
    attempt: int
    error_code: str | None
    skipped: bool = False


@dataclass(frozen=True)
class CallRecord:
    """调用方能看到的观测快照。latency 含退避等待，是用户侧耗时，不是纯上游 RTT。"""

    call_id: str
    alias: str
    provider: str | None
    model: str | None
    stream: bool
    status: CallStatus
    latency_ms: float
    # 流式：从这次调用开始到交出第一条事件的毫秒数，含重试等待。非流式一定是 None。
    # 一条事件都没交出去（失败、取消在首 token 之前、限流）也是 None。
    ttft_ms: float | None
    input_tokens: int
    output_tokens: int
    # attempts：真正打到适配器的次数。
    # retries：同一条路由上因为可重试错误多打的次数，不含第一次，也不含结构化修复。
    attempts: int
    retries: int
    fallback: bool
    error_code: str | None
    # 结构化校验失败后的额外完整调用次数，不含第一次。
    repair_count: int
    attempt_log: tuple[AttemptRecord, ...]


class RecordSink:
    """定长内存队列。超过容量丢掉最旧的，避免长驻进程把记录堆满。"""

    def __init__(
        self,
        *,
        capacity: int,
        on_record: Callable[[CallRecord], None] | None = None,
    ) -> None:
        if capacity < 1:
            raise ValueError("record_capacity 必须 >= 1")
        self._records: deque[CallRecord] = deque(maxlen=capacity)
        self._on_record = on_record
        self._lock = threading.Lock()

    def append(self, record: CallRecord) -> None:
        with self._lock:
            self._records.append(record)
        if self._on_record is None:
            return
        try:
            self._on_record(record)
        except Exception:
            logger.exception("gateway 观测回调失败，不影响本次调用结果")

    def snapshot(self) -> tuple[CallRecord, ...]:
        with self._lock:
            return tuple(self._records)
