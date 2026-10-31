# 步骤 5.5 · Prompt 交给 Langfuse 管理（版本 / label 发布 / 回滚 / 本地兜底）
# 跑：uv run python examples/step05b_langfuse_prompt/main.py
#
# 读代码盯：
#   1) prompts/render.py     — 唯一的 {{var}} 规则 + 严格校验
#   2) prompts/registry.py   — Langfuse 取 production，失败回落本地 .txt
#   3) prompts/scenic_presale.py — 条件逻辑在 Python
#   4) scripts/sync_prompt.py — 本地文件 → Langfuse 新版本

from __future__ import annotations

from dotenv import load_dotenv

from scenic_cs_agent.prompts import render_scenic_presale

load_dotenv()


def show(title: str, **kw) -> None:
    p = render_scenic_presale(**kw)
    print(f"\n=== {title}")
    print(f"source={p.source} version={p.version} fingerprint={p.fingerprint[:12]}")
    print(p.text.split("## 当前能力")[1].split("##")[0].strip())


def main() -> None:
    show("无工具")
    show("有工具", allowed_tools=("query_ticket_price", "query_opening_hours"))
    show("不存在的 label → 兜底", label="no-such-label")


if __name__ == "__main__":
    main()
