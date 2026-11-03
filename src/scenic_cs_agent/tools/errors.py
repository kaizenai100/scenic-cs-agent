"""工具统一错误码。

为什么不直接用字符串报错
------------------------
同一个错误要被三方消费：
  - 模型：根据错误码决定下一步（参数错了就改参数重调；需要审批就告诉用户等待）；
  - 重试策略：只看错误码判断能不能重试，不去解析中文报错文本；
  - 日志 / 监控：按错误码聚合统计（比如 INVALID_ARGS 突然变多，说明 prompt 或 schema 有问题）。
自由文本做不到稳定聚合，所以用枚举。
"""

from __future__ import annotations

from enum import Enum


# 继承 str：序列化成 JSON 时直接就是 "UNKNOWN_TOOL" 这样的字符串，方便写日志和回灌模型。
class ToolErrorCode(str, Enum):
    # 工具没注册，或者注册了但当前场景不可见（对模型统一说"不可用"，不暴露有这个工具）。
    UNKNOWN_TOOL = "UNKNOWN_TOOL"

    # 参数 JSON 解析失败，或 Pydantic 校验不过（缺字段、类型错、格式不匹配）。
    # 模型看到这个码，通常应该修正参数再调一次。
    INVALID_ARGS = "INVALID_ARGS"

    # 参数格式没问题，但业务上不成立（如景区不存在）。区别于 INVALID_ARGS：
    # 这类错误改参数格式没用，应该追问用户或换个说法。
    PRECHECK_FAILED = "PRECHECK_FAILED"

    # 写操作需要人工审批，本次没有执行。不是"失败"，而是"挂起"。
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"

    # 只读工具超时（已经按策略重试过）。只读操作重来一次没有副作用，所以属于可重试错误。
    TIMEOUT = "TIMEOUT"

    # 写操作超时：请求可能已经到达下游并生效了，只是响应没回来。
    # 此时结果是「未知」而不是「失败」，绝对不能自动重放（否则可能重复退款），
    # 应该走查询对账或转人工。
    OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"

    # 工具内部抛了异常（代码 bug、下游返回异常数据等）。
    EXECUTION_ERROR = "EXECUTION_ERROR"


# 「值得重试」的错误码集合。
# 注意：这里只表达"这种错误本身可以重试"，真正是否重试还要看工具是不是只读（见 runtime）。
RETRYABLE = {ToolErrorCode.TIMEOUT}


class ToolFailed(Exception):
    """工具主动失败，文案可以回给模型。

    普通异常只把类型名回给模型，免得把路径或连接信息漏出去。
    业务上说得清的失败（没有这张订单、远端工具返回的错误）用这个异常，
    模型才知道下一步该改参数、追问，还是向用户解释。
    """
