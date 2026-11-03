"""景区 Mock 工具：查开放时间、查票价（只读），提交退票申请（写操作）。

每个工具由三部分组成：
  1) 参数模型（Pydantic）：定义模型要填什么、格式约束是什么；
  2) 执行函数：拿到已校验的参数 + 可信上下文，返回结果 dict；
  3) Tool(...) 注册：声明治理属性（只读/审批/场景/预检查/脱敏）。
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from scenic_cs_agent.tools.base import Tool
from scenic_cs_agent.tools.context import ExecutionContext
from scenic_cs_agent.tools.mock_data import SCENICS


# ======================= 参数模型 =======================


class ScenicArgs(BaseModel):
    """查询类工具的参数：只需要景区名。"""

    # min_length / max_length：挡住空字符串和超长输入（超长可能是注入或模型跑偏）。
    # description 会进入 JSON Schema，模型据此知道该填什么。
    scenic_name: str = Field(min_length=1, max_length=20, description="景区名称，如：云栖山")


class RefundArgs(BaseModel):
    """退票申请的参数。

    注意：这里没有 user_id 字段——"谁在退票"从 ExecutionContext 拿，
    不让模型填，防止越权帮别人退票。
    """

    # pattern 用正则约束格式：O + 8 位数字。模型乱编的订单号在校验阶段就会被拦下。
    order_id: str = Field(pattern=r"^O\d{8}$", description="订单号，如 O20261101")

    # 手机号：11 位、1 开头。属于敏感字段，注册时会声明脱敏。
    phone: str = Field(pattern=r"^1\d{10}$", description="下单手机号")

    reason: str = Field(max_length=100, description="退票原因")


# ======================= 业务预检查 =======================


def scenic_exists(args: BaseModel, ctx: ExecutionContext) -> str | None:
    """预检查：景区是否存在。

    返回 None = 通过；返回字符串 = 拒绝原因（会回给模型）。
    拒绝原因里带上可选景区列表，模型就能据此追问用户"您是想去云栖山还是镜湖湿地？"，
    而不是干巴巴地说"查不到"。
    """
    name = args.scenic_name.strip()
    if name not in SCENICS:
        return f"没有景区「{name}」，可选：{'、'.join(SCENICS)}"
    return None


# ======================= 执行函数 =======================
# 能走到这里，说明白名单、参数校验、预检查都已通过，
# 所以函数里可以直接用 SCENICS[...]，不用再判断景区是否存在。


def get_open_hours(args: ScenicArgs, ctx: ExecutionContext) -> dict[str, Any]:
    """查开放时间：只返回模型回答需要的字段，不把整条记录都丢给模型（省 token、少噪音）。"""
    s = SCENICS[args.scenic_name.strip()]
    return {"open_hours": s["open_hours"], "closed_days": s["closed_days"]}


def get_ticket_price(args: ScenicArgs, ctx: ExecutionContext) -> dict[str, Any]:
    """查票价：带上币种，避免模型自己猜单位。"""
    s = SCENICS[args.scenic_name.strip()]
    return {"currency": "CNY", "tickets": s["tickets"]}


def create_refund_request(args: RefundArgs, ctx: ExecutionContext) -> dict[str, Any]:
    """提交退票申请（Mock）：只生成一张待处理的申请单，不真的退钱。

    关键点：user_id 用的是 ctx.user_id（服务端鉴权得到的），不是模型给的。
    """
    return {"refund_id": f"R{uuid.uuid4().hex[:8]}", "user_id": ctx.user_id, "status": "PENDING"}


# ======================= 注册 =======================

SCENIC_TOOLS = [
    # 查询类：默认只读、无需审批、售前售后都可见；挂上"景区存在"预检查。
    Tool("get_open_hours", "查询景区开放时间和闭园日", ScenicArgs, get_open_hours,
         precheck=scenic_exists),
    Tool("get_ticket_price", "查询景区各类门票价格", ScenicArgs, get_ticket_price,
         precheck=scenic_exists),
    # 写操作：非只读（超时不重试）、需要审批、只在售后场景可见、手机号脱敏。
    Tool("create_refund_request", "为用户提交退票申请", RefundArgs, create_refund_request,
         read_only=False, requires_approval=True, scenes=frozenset({"aftersale"}),
         sensitive_fields=frozenset({"phone"})),
]
