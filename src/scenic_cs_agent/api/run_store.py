"""进程内的 Run 状态和事件日志。

一条 Run 同时被一个生产者和零到多个 SSE 订阅者使用。
事件追加和状态变化都在同一把 Condition 里完成：
  - 改 next_seq / events / status 时不会和别人交错
  - notify_all 把所有正堵在「等下一条」的订阅者叫醒

这只解决单进程。通知丢了也没关系，事件列表还在，
订阅者重连时用 Last-Event-ID 按 seq 把缺的补上。
多进程时再把这份列表换成按 (run_id, seq) 去重的存储即可，接口不用改成透传供应商帧。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from scenic_cs_agent.api.events import (
    RUN_EVENT_TYPES,
    SCHEMA_VERSION,
    RunEvent,
    sanitize_failure,
)


class RunStatus(StrEnum):
    """Run 的生命周期。

    running → cancelling → cancelled 是取消专用的三步，中间态让第二下取消变成空操作。
    completed / failed / cancelled 都是终态，不能再被别的终态盖掉。
    """

    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


TERMINAL_STATUSES: frozenset[RunStatus] = frozenset(
    {
        RunStatus.CANCELLED,
        RunStatus.COMPLETED,
        RunStatus.FAILED,
    }
)


def caught_up_locked(state: RunState, cursor: int) -> bool:
    """订阅者是否已经发完终态前的全部事件。调用方必须正持有 state.changed。

    Condition 用的锁不能重入。订阅循环本身就在锁里，如果这里再 async with 一次，
    会自己把自己卡住，已结束的 Run 再被订阅时连接就不再返回。
    """

    if state.status not in TERMINAL_STATUSES:
        return False
    if not state.events:
        return True
    return state.events[-1].seq <= cursor


def new_run_id() -> str:
    """对外的 run_id。前缀只是为了日志里一眼能认出，不参与鉴权。"""

    return "run_" + uuid4().hex


@dataclass
class RunState:
    """一次 Run 在内存里的全部状态。

    next_seq: 下一条事件要用的序号，从 1 开始。
    events: 已提交的事件，按 seq 递增。订阅者断线后靠它重放。
    cancel_event: 协作式取消开关。生产者在每条模型事件之间看它。
    changed: 既是互斥锁，也是「有新事件或状态变了」的条件变量。
    status: 见 RunStatus。只在持有 changed 时修改。
    task: 后台生产者。取消接口在置位 cancel_event 之后还会 task.cancel()，
          用来打断正堵在 await 上的生产者。
    partial_text: 已经写进 text.delta 的正文，取消时原样放进 run.cancelled。
    saw_text_delta: 是否已经向客户端承诺过正文。用来决定失败能不能在带外重试。
    """

    run_id: str
    next_seq: int = 1
    events: list[RunEvent] = field(default_factory=list)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    changed: asyncio.Condition = field(default_factory=asyncio.Condition)
    status: RunStatus = RunStatus.RUNNING
    task: asyncio.Task[None] | None = None
    partial_text: str = ""
    saw_text_delta: bool = False


class RunStore:
    """run_id → RunState。没有过期淘汰，示例和测试够用。"""

    def __init__(self) -> None:
        self._runs: dict[str, RunState] = {}

    def create(self) -> RunState:
        """登记一条新 Run，状态为 running。此时还没有事件，序号从 1 起。"""

        state = RunState(run_id=new_run_id())
        self._runs[state.run_id] = state
        return state

    def get(self, run_id: str) -> RunState | None:
        """找不到就返回 None，由 HTTP 层决定回 404。"""

        return self._runs.get(run_id)

    def all(self) -> list[RunState]:
        """当前进程里的全部 Run。关闭时用来取消还没结束的任务。"""

        return list(self._runs.values())

    def _append_locked(self, state: RunState, event_type: str, data: dict[str, Any]) -> RunEvent:
        """在已持有 state.changed 的前提下追加一条事件并叫醒订阅者。

        seq 只在这里加，避免两个协程各写各的导致重号或空洞。
        浅拷贝 data：调用方之后如果改自己的 dict，已经入库的事件不变。
        """

        if event_type not in RUN_EVENT_TYPES:
            raise ValueError(f"未知 RunEvent 类型: {event_type}")
        event = RunEvent(
            schema_version=SCHEMA_VERSION,
            run_id=state.run_id,
            seq=state.next_seq,
            type=event_type,
            created_at=datetime.now(timezone.utc).isoformat(),
            data=dict(data),
        )
        state.next_seq += 1
        state.events.append(event)
        if event_type == "text.delta":
            delta = data.get("delta") or ""
            # 空增量不算「客户端已经看见正文」，否则一个空包就会堵死首包前的重试。
            if isinstance(delta, str) and delta:
                state.partial_text += delta
                state.saw_text_delta = True
        state.changed.notify_all()
        return event

    async def append(self, state: RunState, event_type: str, data: dict[str, Any]) -> RunEvent | None:
        """追加一条非终态事件。Run 已经在收尾时返回 None，调用方应改写终态事件。

        取消开关一旦置位就不再收正文：否则取消请求已经返回，后面又冒出 text.delta。
        """

        async with state.changed:
            if state.status != RunStatus.RUNNING or state.cancel_event.is_set():
                return None
            return self._append_locked(state, event_type, data)

    async def complete(
        self,
        state: RunState,
        *,
        finish_reason: str | None,
        usage: dict[str, Any] | None,
    ) -> bool:
        """成功收尾：同一把锁里写入 message.completed 和 run.completed。

        两件事必须原子。如果中间让出锁，取消可能插进来，
        客户端会先看到「消息已完成」再看到「已取消」，语义矛盾。
        已经不在 running（例如取消刚抢先）则一条都不写，返回 False。
        """

        async with state.changed:
            if state.status != RunStatus.RUNNING or state.cancel_event.is_set():
                return False
            reason = finish_reason or "stop"
            self._append_locked(
                state,
                "message.completed",
                {"text": state.partial_text, "finish_reason": reason},
            )
            completed: dict[str, Any] = {"finish_reason": reason}
            if usage is not None:
                completed["usage"] = usage
            self._append_locked(state, "run.completed", completed)
            state.status = RunStatus.COMPLETED
            return True

    async def fail(self, state: RunState, failure: dict[str, Any]) -> bool:
        """写入 run.failed 并进入 failed。只允许从 running 进入。

        失败载荷先过 sanitize_failure：键不对、像异常文本，都换成固定文案。
        不在这里 raise。头一旦开始往外推，再抛异常客户端只能看到连接断开。
        """

        public = sanitize_failure(failure)
        async with state.changed:
            if state.status != RunStatus.RUNNING:
                return False
            self._append_locked(state, "run.failed", public)
            state.status = RunStatus.FAILED
            return True

    async def request_cancel(self, state: RunState) -> bool:
        """协作式取消的第一段：仅 running → cancelling，并置位 cancel_event。

        这是比较并交换：第二次调用、或 Run 已经成功/失败，都返回 False，
        调用方就不要再 task.cancel()，避免把取消计数加到清理逻辑上。
        此处还不写 run.cancelled。正文可能还在飞，等生产者停稳再记 partial_text。
        """

        async with state.changed:
            if state.status != RunStatus.RUNNING:
                return False
            state.status = RunStatus.CANCELLING
            state.cancel_event.set()
            # 订阅者可能正堵在 15 秒心跳上。先叫醒它们，随后生产者补的
            # run.cancelled 会再通知一次；这一次只是少等一会儿。
            state.changed.notify_all()
            return True

    async def mark_cancelled(self, state: RunState, *, reason: str) -> bool:
        """协作式取消的第二段：写下 run.cancelled，状态进入 cancelled。

        running 和 cancelling 都可以进来。前者覆盖「任务被直接 cancel、
        还没来得及走 request_cancel」的路径。已经是终态则不动，保证只有一条终态事件。
        """

        async with state.changed:
            if state.status in TERMINAL_STATUSES:
                return False
            self._append_locked(
                state,
                "run.cancelled",
                {"reason": reason, "partial_text": state.partial_text},
            )
            state.status = RunStatus.CANCELLED
            return True

    async def events_after(self, state: RunState, last_seq: int) -> list[RunEvent]:
        """拷贝 seq 大于 last_seq 的事件。给测试和重放用，订阅循环自己也会取。"""

        async with state.changed:
            return [event for event in state.events if event.seq > last_seq]

    async def caught_up(self, state: RunState, cursor: int) -> bool:
        """订阅者是否已经把终态之前的事件都发出去了。

        还在 running / cancelling 时永远不算追上，否则会把还没写完的 Run 提前关掉。
        """

        async with state.changed:
            return caught_up_locked(state, cursor)
