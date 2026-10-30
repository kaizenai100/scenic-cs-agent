# 步骤 5 · 结构化输出（Pydantic 校验 + 错误回灌修复 + 兜底）
# 跑：uv run python examples/step05_structured/main.py
#
# 读代码盯：
#   1) structured/schemas.py — 输出契约（意图 + 槽位）
#   2) structured/runner.py  — 解析 → 校验 → 带错重试 → 兜底
#   3) 下面 BrokenOnceAdapter — 人为让第一次输出坏掉，看修复链路

from __future__ import annotations

from dotenv import load_dotenv

from scenic_cs_agent.prompts import render_scenic_presale
from scenic_cs_agent.runtime import DeepSeekChatAdapter, ModelRequest, ModelResult
from scenic_cs_agent.structured import classify_intent

load_dotenv()

CASES = [
    "明天想去苏州玩，有什么园子推荐？",
    "拙政园 11 月 2 号两个大人一个小孩，票多少钱？",
    "我昨天买的票能退吗",
]


class BrokenOnceAdapter(DeepSeekChatAdapter):
    """演示用：第一次故意返回缺字段的 JSON，之后正常。"""

    name = "deepseek-chat(broken-once)"

    def __init__(self) -> None:
        super().__init__()
        self._broken = False

    def generate(self, request: ModelRequest) -> ModelResult:
        if not self._broken:
            self._broken = True
            return ModelResult(kind="structured", text='{"intent": "ticket_price"}')
        return super().generate(request)


def show(title: str, outcome) -> None:
    print(f"\n=== {title}")
    for a in outcome.attempts:
        status = "OK" if a.error is None else "FAIL: " + a.error.splitlines()[0]
        print(f"  attempt#{a.index} {status}")
    print(f"  fallback={outcome.used_fallback}")
    print(outcome.result.model_dump_json(indent=2))


def main() -> None:
    system = render_scenic_presale().text
    adapter = DeepSeekChatAdapter()
    for text in CASES:
        show(text, classify_intent(adapter, system=system, user=text))

    show("[修复演示] " + CASES[1], classify_intent(BrokenOnceAdapter(), system=system, user=CASES[1]))


if __name__ == "__main__":
    main()
