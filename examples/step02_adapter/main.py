# 步骤 2 · Model Adapter（Chat / Responses 可切换）
# 跑：
#   uv run python examples/step02_adapter/main.py
#   ADAPTER=responses uv run python examples/step02_adapter/main.py

from __future__ import annotations

import os

from dotenv import load_dotenv

from scenic_cs_agent.runtime import (
    DeepSeekChatAdapter,
    DeepSeekResponsesAdapter,
    ModelRequest,
    OpenAIResponsesAdapter,
)

load_dotenv()

SYSTEM = (
    "你是景区门票售前智能客服助手（教学 Mock）。"
    "只能：理解游玩咨询、追问缺失槽位（景区名/日期/人群）、说明需要查什么。"
    "禁止：编造票价或开放时间；承诺已下单/已退款；在未调用查询工具前给出确定营业数据。"
    "当前还没有工具，若用户问具体票价/开放时间，明确说需要先查景区资料，并追问景区名。"
)

USER = "明天想去苏州玩，有什么园子推荐？门票多少钱？"


def build_adapter():
    kind = os.getenv("ADAPTER", "deepseek").strip().lower()
    if kind in {"responses", "deepseek-responses"}:
        return DeepSeekResponsesAdapter()
    if kind in {"openai", "openai-responses"}:
        return OpenAIResponsesAdapter()
    return DeepSeekChatAdapter()


def main() -> None:
    adapter = build_adapter()
    result = adapter.generate(
        ModelRequest(system=SYSTEM, user=USER, max_output_tokens=512)
    )
    print(f"[adapter={adapter.name}] tokens in/out={result.input_tokens}/{result.output_tokens}")
    print(result.text)


if __name__ == "__main__":
    main()
