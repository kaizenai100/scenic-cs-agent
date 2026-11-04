"""HTTP 请求和响应的形状。

字段注释写给接接口的人看：哪些是客户端能填的，哪些只是服务端状态。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CreateRunRequest(BaseModel):
    """创建一次对话。多出来的字段直接 422，避免客户端拼错还以为生效了。"""

    model_config = ConfigDict(extra="forbid")

    # 用户这句话。空字符串拒绝，过长也拒绝，避免把整段网页塞进来。
    message: str = Field(min_length=1, max_length=4000)
    # 不传就用售前 prompt。传了就整段替换，不再套模板。
    system: str | None = Field(default=None, max_length=8000)
    # 决定这轮能看见哪些工具。售前看不到退票。
    scene: Literal["presale", "aftersale"] = "presale"


class CreateRunResponse(BaseModel):
    """202 的正文。只有 run_id，模型输出在 SSE 里。"""

    model_config = ConfigDict(extra="forbid")

    # 订阅和取消都用这个 id。
    run_id: str


class CancelRunResponse(BaseModel):
    """取消接口的应答。

    accepted 为真表示这次调用把 running 推进到了 cancelling。
    已经结束的 Run 再取消，accepted 为假，status 保持原来的终态。
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    # running / cancelling / cancelled / completed / failed
    status: str
    accepted: bool


class HealthResponse(BaseModel):
    """进程还在。不看模型，也不看 MCP。"""

    model_config = ConfigDict(extra="forbid")

    status: str = "ok"


class McpServerView(BaseModel):
    """一台 MCP 服务现在的连接情况。"""

    model_config = ConfigDict(extra="forbid")

    # connected：已连上；disabled：配置里关掉了；error：非必需服务没连上
    state: str
    # 失败说明。连上或未启用时是空字符串。
    error: str = ""


class ReadyzResponse(BaseModel):
    """就绪检查。进程已经起来才会返回这份；必需依赖失败时进程根本不会监听。"""

    model_config = ConfigDict(extra="forbid")

    # ok：熔断都是 closed，MCP 也没有错误。degraded：还能应答，但有熔断打开或 MCP 报错。
    status: str
    # provider 名 → closed / open / half_open。这是上游名字，不是模型别名。
    gateway: dict[str, str]
    # MCP 服务名 → 连接情况。
    mcp: dict[str, McpServerView]
    # 非必需服务的启动错误。必需服务失败不会出现在这里，因为那样进程起不来。
    errors: list[str]
