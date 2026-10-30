"""售前意图识别的输出契约。

模型必须按这个结构回：意图 + 槽位 + 缺失项 + 追问。
字段约束写在 Pydantic 里，校验失败的报错会原样回传给模型去修。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal[
    "ticket_price",  # 问票价
    "opening_hours",  # 问开放时间
    "recommend",  # 求推荐
    "booking",  # 想下单
    "refund",  # 想退票
    "chitchat",  # 闲聊
    "unknown",
]


class Slots(BaseModel):
    scenic_name: str | None = Field(None, description="景区名，如 拙政园")
    visit_date: str | None = Field(None, description="游玩日期，YYYY-MM-DD 或用户原话")
    visitors: str | None = Field(None, description="人群，如 2 成人 1 儿童")


class IntentResult(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0, le=1)
    slots: Slots
    missing_slots: list[Literal["scenic_name", "visit_date", "visitors"]]
    reply: str = Field(min_length=1, max_length=300, description="给用户的简短回复/追问")
