"""Run 级事件信封。

为什么单独有这一层：
  适配器的 ModelStreamEvent 只回答「模型这一次调用发生了什么」
  （一段正文、结束原因、token 用量）。浏览器要看的是「这一次用户 Run」：
  开始了、正文增量、这条消息写完、整次成功 / 失败 / 取消。
  model.finished 不等于 run.completed：finish_reason 为 length 或
  content_filter 时，模型是结束了，但业务上算失败。

信封固定为
  RunEvent{schema_version, run_id, seq, type, created_at, data}。
  前端不认识的 type 应忽略，这样以后加事件不必先改所有客户端。
  服务端只发出下面这份名单里的类型，避免把供应商字段顺手塞进 data。
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# 协议版本写在每条事件上。以后不兼容的改动递增它，客户端可以拒绝旧形状。
SCHEMA_VERSION = "1"

# 数据面、过程面、控制面都列在这里。
# tool.* / run.retrying 允许写入，但默认生产者不挂工具循环。
RUN_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "run.started",
        "text.delta",
        "message.completed",
        "tool.started",
        "tool.completed",
        "run.retrying",
        "run.completed",
        "run.failed",
        "run.cancelled",
    }
)

# 这三种一出现，订阅就可以结束：Run 不会再有新的业务事件。
TERMINAL_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "run.completed",
        "run.failed",
        "run.cancelled",
    }
)

# 失败事件的 data 只允许这四个键。多一个键就有机会把堆栈、提示词、密钥带出去。
FAILURE_DATA_KEYS: frozenset[str] = frozenset({"code", "stage", "retryable", "hint"})

_TOKEN = re.compile(r"^[a-z0-9_.-]{1,40}$")

# 任何看起来不干净的失败都收成这一条。hint 是写死的中文，不拼接异常文本。
GENERIC_FAILURE: dict[str, Any] = {
    "code": "internal",
    "stage": "generate",
    "retryable": False,
    "hint": "生成失败，请稍后重试",
}


class RunEvent(BaseModel):
    """一条已经编号、可以重放的 Run 事件。

    seq 从 1 起单调递增，由 RunStore 分配，调用方不许自己填。
    created_at 用带时区的 ISO 字符串，编码成 JSON 时不再做二次转换。
    data 的形状由 type 决定；本模型故意不按 type 拆成多个子类，
    这样 SSE 编码和存储可以只认一种信封。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=SCHEMA_VERSION)
    run_id: str
    seq: int = Field(ge=1)
    type: str
    created_at: str
    data: dict[str, Any]


def tool_started_data(tool_call_id: str, name: str) -> dict[str, str]:
    """工具开始时的 data。Loop 接上之后用它配对同一次调用。

    本步没有执行工具，只把形状定下来，避免以后各处手写字段名。
    tool_call_id 必须是调用方生成的稳定 id，不能用列表下标，
    否则并行或重试时对不上 tool.completed。
    """

    return {"tool_call_id": tool_call_id, "name": name}


def tool_completed_data(tool_call_id: str, name: str, *, ok: bool) -> dict[str, Any]:
    """工具结束时的 data。ok 只表示这次调用有没有正常返回，不携带原始输出。

    原始输出可能很大，也可能含内部数据。要给用户看的摘要由 Loop 另选字段，
    不在这个最小形状里塞。
    """

    return {"tool_call_id": tool_call_id, "name": name, "ok": ok}


def sanitize_failure(raw: dict[str, Any]) -> dict[str, Any]:
    """把失败收成可以给客户端的四个字段。

    不接受调用方拼出来的长文本：异常消息里经常有上游 URL，
    查询串里可能夹着密钥。不合规则就换成固定文案。
    """

    code = raw.get("code")
    stage = raw.get("stage")
    retryable = raw.get("retryable")
    hint = raw.get("hint")
    if not isinstance(code, str) or _TOKEN.fullmatch(code) is None:
        return dict(GENERIC_FAILURE)
    if not isinstance(stage, str) or _TOKEN.fullmatch(stage) is None:
        return dict(GENERIC_FAILURE)
    if not isinstance(retryable, bool):
        return dict(GENERIC_FAILURE)
    if not isinstance(hint, str) or not hint.strip() or len(hint) > 200:
        return dict(GENERIC_FAILURE)
    # 换行会把 SSE 的 data 行拆碎；这里直接拒绝，而不是替换掉换行再发出去。
    if "\n" in hint or "\r" in hint:
        return dict(GENERIC_FAILURE)
    if set(raw) != FAILURE_DATA_KEYS:
        return dict(GENERIC_FAILURE)
    return {"code": code, "stage": stage, "retryable": retryable, "hint": hint}
