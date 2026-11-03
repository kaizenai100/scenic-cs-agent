"""工具契约：一个工具 = 名字 + 说明 + 参数模型 + 执行函数 + 治理属性。

核心思路
--------
1) 参数用 Pydantic 定义，一份定义两处用：
   - 生成给模型看的 JSON Schema（function calling 的 parameters）；
   - 执行前校验模型给的参数（模型输出不可信，可能缺字段、类型错、乱填）。
   两处用同一份定义，就不会出现"告诉模型的格式"和"实际校验的格式"不一致。

2) 「治理属性」声明在工具上，执行逻辑统一在 ToolRuntime 里：
   工具作者只需声明"我是只读的 / 我要审批 / 我超时 3 秒"，
   不用每个工具自己写超时、重试、审计——这些横切逻辑集中一处，不会漏。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from scenic_cs_agent.tools.context import ExecutionContext
from scenic_cs_agent.tools.errors import ToolErrorCode

# 工具函数签名：(已校验的参数对象, 可信上下文) -> 结果 dict
#   - 第一个参数已经是 Pydantic 对象，工具里不用再判空、判类型；
#   - 第二个参数是服务端注入的上下文，需要身份就从这里拿，不从参数拿。
ToolFunc = Callable[[BaseModel, ExecutionContext], dict[str, Any]]

# 业务预检查函数签名：返回 None 表示通过；返回字符串表示拒绝原因（会回给模型）。
# 放在执行前做：格式对不代表业务成立，例如景区名格式没问题，但根本没这个景区。
PrecheckFunc = Callable[[BaseModel, ExecutionContext], str | None]


# frozen=True：工具定义注册后不可变，防止运行中被意外改掉超时、审批等属性。
@dataclass(frozen=True)
class Tool:
    # 工具名：模型通过它"点名"要调哪个工具，必须唯一，建议用动词_名词的蛇形命名。
    name: str

    # 工具说明：会原样发给模型，模型靠它判断"什么时候该用这个工具"。
    # 写清楚能做什么、不能做什么，比写得花哨更重要。
    description: str

    # 参数模型（Pydantic 类）：既生成 schema，又做校验。
    args_model: type[BaseModel]

    # 真正干活的函数。
    func: ToolFunc

    # ---------- 以下是治理属性，由 ToolRuntime 统一执行 ----------

    # 是否只读。只读操作（查询）重复执行没有副作用，超时可以重试；
    # 写操作（下单、退款）重复执行可能造成重复扣款/重复退款，超时不能自动重试。
    read_only: bool = True

    # 硬超时（秒）。超过就不再等，防止一个慢工具拖死整轮对话。
    timeout_s: float = 3.0

    # 是否需要人工审批。为 True 时，第一次调用只返回 APPROVAL_REQUIRED，不执行。
    requires_approval: bool = False

    # 在哪些场景下可见。不在列表里的场景，模型既看不到这个工具，点名了也执行不了。
    scenes: frozenset[str] = frozenset({"presale", "aftersale"})

    # 业务预检查（可选）。
    precheck: PrecheckFunc | None = None

    # 需要脱敏的参数字段名（如手机号、身份证）。审计日志和审批预览里会打码。
    # 用 field(default_factory=...) 是 dataclass 对可变/容器默认值的惯用写法。
    sensitive_fields: frozenset[str] = field(default_factory=frozenset)

    def to_openai_schema(self) -> dict[str, Any]:
        """转成 chat.completions 的 tools 数组元素。

        格式是 OpenAI 定义的，DeepSeek 兼容同样格式：
          {"type": "function", "function": {"name", "description", "parameters"}}
        parameters 直接用 Pydantic 自动生成的 JSON Schema，
        字段上的 Field(description=...) 会变成 schema 里的 description，模型能看到。
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


@dataclass(frozen=True)
class ToolResult:
    """工具执行结果。

    设计原则：失败也是结果。
    工具出错时不抛异常打断整个请求，而是返回 ok=False + 错误码，
    这样到了步骤 7 的 Loop 里，错误可以回灌给模型，让模型自己决定：
    改参数重试、换个工具、还是向用户解释。
    """

    # 对应模型返回的 tool_call.id。回灌给模型时必须带上，
    # 模型靠它把"哪个结果"对应到"哪次调用"（一轮里可能并行调多个工具）。
    tool_call_id: str

    # 工具名，方便日志和调试。
    name: str

    # 是否成功执行并拿到业务结果。
    ok: bool

    # 成功时是业务数据；失败时是 {"error": "给模型看的说明", ...}。
    content: dict[str, Any]

    # 失败时的统一错误码；成功时为 None。
    error_code: ToolErrorCode | None = None

    # 这个错误本身是否可重试（给上层参考，例如 Loop 决定要不要再试）。
    retryable: bool = False

    # 实际尝试了几次（含重试）。审计和排障用：attempts=2 说明第一次超时了。
    attempts: int = 1

    # 整个 execute 的耗时（毫秒），含校验、重试。
    latency_ms: int = 0

    def to_model_payload(self) -> dict[str, Any]:
        """生成回灌给模型的内容。

        只给模型需要的：成功给数据，失败给错误码 + 说明。
        attempts、latency_ms 这类内部指标不给模型，它用不上，还白占 token。
        """
        if self.ok:
            return {"ok": True, **self.content}
        return {"ok": False, "error_code": self.error_code.value, **self.content}
