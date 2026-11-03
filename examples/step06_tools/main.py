"""步骤 6：工具（单轮 function calling + 运行时治理，还不是 Loop）。

本示例分三段：
  1) 真模型：给模型挂上「售前场景可见」的工具，看它会不会、怎么点名工具；
  2) 防御演示：不走模型，直接构造各种"坏调用"喂给 ToolRuntime，看每一关怎么拦；
  3) 超时演示：用两个故意很慢的工具，对比只读超时（会重试）和写操作超时（结果未知）。

本步只到"执行工具"为止。把工具结果回灌给模型、让它基于结果生成最终回答，
并在需要时继续调工具——这个循环是步骤 7 Agent Loop 的内容。

运行：uv run python examples/step06_tools/main.py
"""

import logging
import time

from dotenv import load_dotenv
from pydantic import BaseModel

from scenic_cs_agent.prompts import render_scenic_presale
from scenic_cs_agent.runtime.deepseek_chat_adapter import DeepSeekChatAdapter
from scenic_cs_agent.runtime.model_types import ModelRequest
from scenic_cs_agent.tools import SCENIC_TOOLS, ExecutionContext, Tool, ToolRuntime

# 读取 .env 里的 DEEPSEEK_API_KEY、LANGFUSE_* 等配置。
load_dotenv()

# 打开 INFO 日志，让 ToolRuntime 的审计日志打印出来（每次调用一行 JSON）。
# 注意 httpx 的请求日志也会用同一格式打印，前缀 [audit] 只是格式，不代表都是审计日志。
logging.basicConfig(level=logging.INFO, format="  [audit] %(message)s")

# 两个可信上下文：真实系统里由服务端在鉴权后构造，这里手写模拟。
# 同一个用户，分别处在售前和售后两个场景。
presale = ExecutionContext(run_id="run-001", user_id="u-1001", channel="app", scene="presale")
aftersale = ExecutionContext(run_id="run-002", user_id="u-1001", channel="app", scene="aftersale")


def show(r):
    """打印一次执行结果：是否成功、错误码、尝试次数，以及会回灌给模型的内容。"""
    print(f"  ok={r.ok} code={r.error_code and r.error_code.value} attempts={r.attempts} → {r.to_model_payload()}")


# ======================= 1. 真模型 =======================
runtime = ToolRuntime(SCENIC_TOOLS)
adapter = DeepSeekChatAdapter()

# 对比两个场景能看到的工具：售前没有退票工具。
print("售前可见工具：", runtime.names(presale))
print("售后可见工具：", runtime.names(aftersale))

# system prompt 里"已挂载工具"那段，用的也是当前场景可见的工具名，和发给模型的 tools 保持一致。
system = render_scenic_presale(allowed_tools=runtime.names(presale)).text

for q in [
    "云栖山几点开门？学生票多少钱？",  # 期望：模型一次返回两个 tool_calls（并行调用）
    "你们这有什么好玩的？",  # 期望：信息不足，不调工具，直接追问景区名
]:
    print(f"\n=== 用户：{q}")
    result = adapter.generate(ModelRequest(system=system, user=q, tools=runtime.schemas(presale)))
    print(f"kind={result.kind} finish={result.finish_reason}")
    if result.kind != "tool_calls":
        print("模型直接回答：", result.text)
        continue
    # 模型只是"点名"，真正执行由 ToolRuntime 层层把关。
    for call in result.tool_calls:
        print(f"→ {call['name']}({call['arguments']})")
        show(runtime.execute(call, presale))

# ======================= 2. 防御演示 =======================
# 不走模型，直接构造调用，确保每一关都能被稳定触发、可重复验证。
print("\n=== 防御演示")
refund = '{"order_id":"O20261101","phone":"13812345678","reason":"行程变更"}'
cases = [
    # 模型编了一个根本不存在的工具名 → 执行期白名单拦截
    ("未注册工具", {"id": "x1", "name": "book_ticket", "arguments": "{}"}, presale),
    # 工具存在，但售前场景不可见（比如被注入诱导） → 同样按 UNKNOWN_TOOL 拦截
    ("跨场景越权", {"id": "x2", "name": "create_refund_request", "arguments": refund}, presale),
    # 参数不是合法 JSON → INVALID_ARGS
    ("JSON 坏了", {"id": "x3", "name": "get_ticket_price", "arguments": "{oops"}, presale),
    # JSON 合法但缺必填字段 → Pydantic 校验失败，INVALID_ARGS
    ("缺必填参数", {"id": "x4", "name": "get_ticket_price", "arguments": "{}"}, presale),
    # 格式完全正确，但景区不存在 → 业务预检查拦截，PRECHECK_FAILED
    ("业务预检查", {"id": "x5", "name": "get_ticket_price", "arguments": '{"scenic_name":"泰山"}'}, presale),
    # 售后场景可见，但写操作需要审批 → 不执行，返回脱敏的参数预览
    ("写操作待审批", {"id": "x6", "name": "create_refund_request", "arguments": refund}, aftersale),
]
for label, call, ctx in cases:
    print(f"- {label}")
    show(runtime.execute(call, ctx))

# 模拟审批通过后再执行同一调用：结果里的 user_id 来自 ctx，模型参数里压根没有 user_id。
print("- 审批通过后执行（user_id 来自上下文，不来自模型）")
show(runtime.execute(cases[-1][1], aftersale, approved=True))


# ======================= 3. 超时演示 =======================
class NoArgs(BaseModel):
    """无参工具的参数模型：空模型，arguments 传 "{}" 即可。"""


def slow(args, ctx):
    """模拟下游接口很慢：睡 1 秒，远超下面设置的 0.2 秒超时。"""
    time.sleep(1)
    return {"done": True}


# 单独建一个 runtime，只放两个慢工具，避免影响上面的真实工具。
slow_rt = ToolRuntime([
    Tool("slow_query", "慢查询", NoArgs, slow, timeout_s=0.2),  # 只读（默认）
    Tool("slow_write", "慢写入", NoArgs, slow, timeout_s=0.2, read_only=False),  # 写操作
])
print("\n=== 超时演示")
print("- 只读超时：自动重试 1 次后放弃（attempts=2，耗时约 0.4s）")
show(slow_rt.execute({"id": "t1", "name": "slow_query", "arguments": "{}"}, presale))
print("- 写操作超时：不重试，标记结果未知（attempts=1）")
show(slow_rt.execute({"id": "t2", "name": "slow_write", "arguments": "{}"}, presale))
