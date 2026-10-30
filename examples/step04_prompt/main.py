# 步骤 4 · Prompt 模板化（Jinja2 + StrictUndefined）
# 跑：uv run python examples/step04_prompt/main.py
#
# 读代码盯：
#   1) prompts/scenic_presale_v1.md.j2 — 行为契约写在模板里
#   2) render_scenic_presale() — StrictUndefined + fingerprint
#   3) main 不再手写一大段 SYSTEM 字符串

from __future__ import annotations

from dotenv import load_dotenv

from scenic_cs_agent.prompts import render_scenic_presale
from scenic_cs_agent.runtime import DeepSeekChatAdapter, ModelRequest

load_dotenv()

USER = "明天想去苏州玩，有什么园子推荐？门票多少钱？"


def main() -> None:
    prompt = render_scenic_presale(tools_enabled=False)
    print(f"[prompt] version={prompt.version} template={prompt.template_name}")
    print(f"[prompt] fingerprint={prompt.fingerprint[:12]}...")
    print("--- system preview ---")
    print(prompt.text)
    print("--- model ---")

    adapter = DeepSeekChatAdapter()
    result = adapter.generate(
        ModelRequest(system=prompt.text, user=USER, max_output_tokens=512)
    )
    print(f"[adapter={adapter.name}] tokens in/out={result.input_tokens}/{result.output_tokens}")
    print(result.text)


if __name__ == "__main__":
    main()
