# 步骤 3 · Streaming（Chat Adapter.stream → ModelStreamEvent）
# 跑：uv run python examples/step03_stream/main.py
# 对照整段：MODE=once uv run python examples/step03_stream/main.py

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from scenic_cs_agent.runtime import DeepSeekChatAdapter, ModelRequest

load_dotenv()

SYSTEM = (
    "你是景区门票售前智能客服助手（教学 Mock）。"
    "只能：理解游玩咨询、追问缺失槽位（景区名/日期/人群）、说明需要查什么。"
    "禁止：编造票价或开放时间；承诺已下单/已退款；在未调用查询工具前给出确定营业数据。"
    "当前还没有工具，若用户问具体票价/开放时间，明确说需要先查景区资料，并追问景区名。"
)

USER = "明天想去苏州玩，有什么园子推荐？门票多少钱？"


def main() -> None:
    adapter = DeepSeekChatAdapter()
    request = ModelRequest(system=SYSTEM, user=USER, max_output_tokens=512)
    mode = os.getenv("MODE", "stream").strip().lower()

    if mode == "once":
        result = adapter.generate(request)
        print(
            f"[adapter={adapter.name} mode=once] "
            f"tokens in/out={result.input_tokens}/{result.output_tokens}"
        )
        print(result.text)
        return

    print(f"[adapter={adapter.name} mode=stream]", flush=True)
    for event in adapter.stream(request):
        if event.type == "text.delta":
            print(event.data["delta"], end="", flush=True)
        elif event.type == "model.finished":
            print(f"\n[finished] {event.data.get('finish_reason')}", flush=True)
        elif event.type == "model.usage":
            u = event.data
            print(
                f"[usage] in/out/total="
                f"{u.get('input_tokens')}/{u.get('output_tokens')}/{u.get('total_tokens')}",
                flush=True,
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[cancelled by user]", file=sys.stderr)
        raise SystemExit(130) from None
