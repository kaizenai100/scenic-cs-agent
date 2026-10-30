# 步骤 1 · 裸 LLM 调用（Chat Completions，无 Adapter）
# 跑：uv run python examples/step01_llm/main.py

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

messages = [
    {
        "role": "system",
        "content": (
            "你是景区门票售前智能客服助手（教学 Mock）。"
            "只能：理解游玩咨询、追问缺失槽位（景区名/日期/人群）、说明需要查什么。"
            "禁止：编造票价或开放时间；承诺已下单/已退款；在未调用查询工具前给出确定营业数据。"
            "当前还没有工具，若用户问具体票价/开放时间，明确说需要先查景区资料，并追问景区名。"
        ),
    },
    {
        "role": "user",
        "content": "明天想去苏州玩，有什么园子推荐？门票多少钱？",
    },
]

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
)

print(resp.choices[0].message.content)
