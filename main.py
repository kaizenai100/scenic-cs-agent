# 根入口只做导航：每一步有独立 examples/stepXX_*/main.py
# 当前请跑步骤 8（进程内 LLM Gateway，Fake 不需要 Key）：
#   uv run python examples/step08_llm_gateway/main.py
#
# 回顾：
#   uv run python examples/step05_structured/main.py
#   uv run python examples/step07_agent_loop/main.py

print(
    "请按步骤跑独立入口，例如：\n"
    "  uv run python examples/step08_llm_gateway/main.py\n"
    "目录见 examples/"
)
