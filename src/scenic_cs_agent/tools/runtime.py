"""ToolRuntime：注册表 + 安全执行。

执行一次工具调用依次过这几关：
  1) 可见性白名单：schemas(ctx) 只把当前场景可见的工具给模型
  2) 执行期白名单：execute 时再查一次（模型可能编名字，也可能点名别的场景的工具）
  3) 参数校验：JSON 解析 + Pydantic 校验
  4) 业务预检查：格式对不代表业务成立
  5) 审批：写操作先返回 APPROVAL_REQUIRED，不执行
  6) 硬超时 + 重试：只读工具超时可重试；写操作超时 = 结果未知，绝不自动重放
  7) 异常收口 + 审计日志（敏感字段脱敏）
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any

from pydantic import BaseModel, ValidationError

from scenic_cs_agent.tools.base import Tool, ToolResult
from scenic_cs_agent.tools.context import ExecutionContext
from scenic_cs_agent.tools.errors import RETRYABLE, ToolErrorCode

audit_log = logging.getLogger("scenic_cs_agent.tool_audit")


def _mask(value: Any) -> str:
    s = str(value)
    return s[:3] + "****" + s[-2:] if len(s) > 5 else "****"


class ToolRuntime:
    def __init__(self, tools: list[Tool], *, max_read_retries: int = 1) -> None:
        self._tools = {t.name: t for t in tools}
        self._max_read_retries = max_read_retries
        # 共享线程池跑工具，以便设硬超时（超时后线程可能还在跑，结果直接丢弃）
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool")

    def visible_tools(self, ctx: ExecutionContext) -> list[Tool]:
        return [t for t in self._tools.values() if ctx.scene in t.scenes]

    def names(self, ctx: ExecutionContext) -> tuple[str, ...]:
        return tuple(t.name for t in self.visible_tools(ctx))

    def schemas(self, ctx: ExecutionContext) -> list[dict[str, Any]]:
        return [t.to_openai_schema() for t in self.visible_tools(ctx)]

    def execute(
        self, call: dict[str, Any], ctx: ExecutionContext, *, approved: bool = False
    ) -> ToolResult:
        """call 形如 {"id": ..., "name": ..., "arguments": "<json 字符串>"}。"""
        start = time.monotonic()
        result = self._execute(call, ctx, approved=approved)
        result = ToolResult(**{**result.__dict__, "latency_ms": int((time.monotonic() - start) * 1000)})
        self._audit(call, ctx, result)
        return result

    def _execute(self, call: dict[str, Any], ctx: ExecutionContext, *, approved: bool) -> ToolResult:
        call_id, name = call["id"], call["name"]

        def fail(code: ToolErrorCode, msg: str, attempts: int = 1) -> ToolResult:
            return ToolResult(call_id, name, False, {"error": msg}, code, code in RETRYABLE, attempts)

        tool = self._tools.get(name)
        if tool is None or ctx.scene not in tool.scenes:
            return fail(ToolErrorCode.UNKNOWN_TOOL, f"当前不可用的工具：{name}")

        try:
            args = tool.args_model.model_validate(json.loads(call.get("arguments") or "{}"))
        except (json.JSONDecodeError, ValidationError) as e:
            return fail(ToolErrorCode.INVALID_ARGS, f"参数不合法：{e}")

        if tool.precheck and (reason := tool.precheck(args, ctx)):
            return fail(ToolErrorCode.PRECHECK_FAILED, reason)

        if tool.requires_approval and not approved:
            return ToolResult(call_id, name, False,
                              {"error": "该操作需人工审批，已提交待审", "args_preview": self._redact(tool, args)},
                              ToolErrorCode.APPROVAL_REQUIRED)

        max_attempts = 1 + (self._max_read_retries if tool.read_only else 0)
        for attempt in range(1, max_attempts + 1):
            future = self._pool.submit(tool.func, args, ctx)
            try:
                return ToolResult(call_id, name, True, future.result(timeout=tool.timeout_s), attempts=attempt)
            except FutureTimeout:
                if not tool.read_only:
                    # 写操作超时：对方可能已经执行成功，重放可能重复退款
                    return fail(ToolErrorCode.OUTCOME_UNKNOWN, "操作超时，结果未知，请勿重复提交", attempt)
                if attempt == max_attempts:
                    return fail(ToolErrorCode.TIMEOUT, f"查询超时（{tool.timeout_s}s）", attempt)
            except Exception as e:  # noqa: BLE001 工具内部错误统一收口
                return fail(ToolErrorCode.EXECUTION_ERROR, f"工具执行失败：{type(e).__name__}", attempt)
        raise AssertionError("unreachable")

    @staticmethod
    def _redact(tool: Tool, args: BaseModel) -> dict[str, Any]:
        data = args.model_dump()
        return {k: (_mask(v) if k in tool.sensitive_fields else v) for k, v in data.items()}

    def _audit(self, call: dict[str, Any], ctx: ExecutionContext, r: ToolResult) -> None:
        tool = self._tools.get(call["name"])
        try:
            raw = json.loads(call.get("arguments") or "{}")
            args = {k: (_mask(v) if tool and k in tool.sensitive_fields else v) for k, v in raw.items()}
        except (json.JSONDecodeError, AttributeError):
            args = "<unparseable>"
        audit_log.info(json.dumps({
            "run_id": ctx.run_id, "user_id": ctx.user_id, "scene": ctx.scene,
            "tool": call["name"], "call_id": call["id"], "args": args,
            "ok": r.ok, "error_code": r.error_code.value if r.error_code else None,
            "attempts": r.attempts, "latency_ms": r.latency_ms,
        }, ensure_ascii=False))
