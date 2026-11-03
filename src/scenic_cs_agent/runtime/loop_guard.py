"""LoopGuard：任务内 Agent Loop 的安全护栏。

四层能力里，本模块只负责最外这一层「护栏」：
  1) Function Calling —— 模型用协议点名工具（Adapter 归一）
  2) Tool Runtime    —— 白名单 / 校验 / 审批 / 超时（tools/runtime.py）
  3) Agent Loop      —— while 调模型 → 执行工具 → 回灌（agent_loop.py）
  4) LoopGuard       —— 限制轮数、拦截重复动作、响应取消（本文件）

为什么需要护栏
--------------
模型可能陷入「反复调同一个工具」的死循环，或一次请求拖几十轮烧 token。
护栏不替代业务完成判断：它只做「协议层 / 资源层」的硬停止。
业务是否答完（Completion Contract）留给后续步骤；本步只在注释里留钩子。
"""

from __future__ import annotations

import json
import threading
from typing import Any


def fingerprint_action(name: str, arguments: str | dict[str, Any] | None) -> str:
    """生成「动作指纹」：工具名 + 归一化后的参数。

    归一化规则：
      - arguments 若是 JSON 字符串，先 loads；解析失败则按原始字符串；
      - dict 用 sort_keys 再 dumps，让 {"a":1,"b":2} 与 {"b":2,"a":1} 视为同一动作。
    这样模型两次「查云栖山票价」即使用不同参数键序，也会被当成重复。
    """
    if isinstance(arguments, dict):
        payload: Any = arguments
    else:
        raw = arguments or "{}"
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            payload = raw
    if isinstance(payload, dict):
        normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    else:
        normalized = str(payload)
    return f"{name}:{normalized}"


class LoopGuard:
    """任务内循环护栏：最大轮数 + 重复动作指纹 + 取消信号。

    用法（由 AgentLoop 驱动，不要在业务代码里自己拼）：
      1. 每轮开始前 check_cancel / check_max_turns；
      2. 模型返回 tool_calls 后，对每个 call 算指纹：
         - 若 should_stop_duplicate → 整轮停止（避免死循环）；
         - 否则 record_action 记入已见集合，再交给 ToolRuntime 执行。
    """

    def __init__(self, *, max_turns: int = 8) -> None:
        if max_turns < 1:
            raise ValueError("max_turns 至少为 1")
        # 允许模型「想问题 → 调工具 → 再回答」的上限轮数（一次 generate = 一轮）。
        self.max_turns = max_turns
        # 已经走过几轮（含因 tool_calls 继续的轮）。
        self.turns_used = 0
        # 本任务内已见过的动作指纹集合；同一指纹再次出现即判定重复。
        self._seen_actions: set[str] = set()

    def check_cancel(self, cancel_event: threading.Event | None) -> str | None:
        """每轮开头查一次取消。返回 stop_reason；None 表示继续。

        cancel_event 由外层（以后的 FastAPI / SSE 取消）注入；
        本步示例可手动 set() 验证路径。不用轮询忙等——Event.is_set() 是非阻塞读。
        """
        if cancel_event is not None and cancel_event.is_set():
            return "cancelled"
        return None

    def begin_turn(self) -> str | None:
        """进入新一轮：计数 +1；若已达上限返回 max_turns。

        放在「调模型之前」调用：这样 max_turns=1 时模型还能跑一轮，
        若该轮仍返回 tool_calls，下一轮 begin_turn 才会拦下——
        验收「有工具调用时也能被轮数护栏停住」正是这个语义。
        """
        self.turns_used += 1
        if self.turns_used > self.max_turns:
            return "max_turns"
        return None

    def record_action(self, name: str, arguments: str | dict[str, Any] | None) -> str:
        """记录一次即将执行的动作，返回其指纹（便于日志）。"""
        fp = fingerprint_action(name, arguments)
        self._seen_actions.add(fp)
        return fp

    def should_stop_duplicate(self, name: str, arguments: str | dict[str, Any] | None) -> bool:
        """若该动作指纹已出现过，返回 True（调用方应停止循环）。

        注意：先 should_stop_duplicate，再决定是否 record_action；
        否则第一次调用也会被自己刚写入的指纹误伤。
        """
        return fingerprint_action(name, arguments) in self._seen_actions

    def reset(self) -> None:
        """清空计数与指纹。单测 / 示例复用同一 Guard 实例时用。"""
        self.turns_used = 0
        self._seen_actions.clear()
