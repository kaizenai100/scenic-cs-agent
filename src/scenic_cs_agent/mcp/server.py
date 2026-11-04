"""景区门票的 MCP 服务（虚构数据）。

给本地把「MCP 服务 → 客户端 → ToolRuntime」这条链路跑通，不连真实票务系统。
用官方 SDK 的 MCPServer，传输走 stdio：stdout 只留给协议，日志写到 stderr。

启动：
    python -m scenic_cs_agent.mcp.server
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from scenic_cs_agent.tools.mock_data import SCENICS

# 虚构订单。订单号规则和本地退票工具一致：O + 8 位数字。
ORDERS: dict[str, dict[str, str]] = {
    "O20261101": {
        "scenic_name": "云栖山",
        "status": "已支付",
        "ticket": "成人票",
        "visit_date": "2026-11-08",
    },
    "O20261102": {
        "scenic_name": "镜湖湿地",
        "status": "已使用",
        "ticket": "学生票",
        "visit_date": "2026-11-02",
    },
}


class OrderView(BaseModel):
    """查单结果。用模型而不是裸 dict，客户端能拿到 structured_content。"""

    order_id: str
    scenic_name: str
    status: str
    ticket: str
    visit_date: str


class OpeningHoursView(BaseModel):
    scenic_name: str
    open_hours: str
    closed_days: list[str]


# stdio 下 SDK 自己会配日志；这里把级别压到 WARNING，避免把协议通道刷脏。
server = MCPServer("scenic-ticket", log_level="WARNING")


@server.tool()
def query_ticket_order(
    order_id: Annotated[str, Field(pattern=r"^O\d{8}$", description="订单号，如 O20261101")],
) -> OrderView:
    """按订单号查询门票订单状态。只查虚构订单，不退款、不改单。"""
    order = ORDERS.get(order_id)
    if order is None:
        raise ToolError(f"没有订单 {order_id}")
    return OrderView(order_id=order_id, **order)


@server.tool()
def get_opening_hours(
    scenic_name: Annotated[str, Field(min_length=1, max_length=20, description="景区名称，如云栖山")],
) -> OpeningHoursView:
    """查询景区开放时间和闭园日。数据是虚构的。"""
    scenic = SCENICS.get(scenic_name.strip())
    if scenic is None:
        known = "、".join(SCENICS)
        raise ToolError(f"没有景区「{scenic_name}」，可选：{known}")
    return OpeningHoursView(
        scenic_name=scenic_name.strip(),
        open_hours=scenic["open_hours"],
        closed_days=list(scenic["closed_days"]),
    )


def _write_pidfile() -> None:
    """SCENIC_MCP_PIDFILE 有值时把 pid 写进去，方便确认子进程已经退出。平时不设置。"""
    path = os.environ.get("SCENIC_MCP_PIDFILE")
    if not path:
        return
    Path(path).write_text(str(os.getpid()), encoding="utf-8")


def main() -> None:
    # stdout 是 JSON-RPC，不能拿来打日志。
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format="%(name)s %(levelname)s %(message)s")
    _write_pidfile()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
