# scenic-cs-agent

开源地址：https://github.com/kaizenai100/scenic-cs-agent

个人练习仓：对照同程 `ai.agent.intelligentcustomer` **只读架构**，从零演进重写一版「景区智能客服」骨架。

## 边界

- 同程仓库：只读对照，不拷贝业务代码 / 密钥 / 客户数据
- 本仓：Mock 工具与假数据；可写进个人 GitHub
- 飞书退款练习：暂搁置（原文档留在 `docs/archive-feishu-refund/`）

## 怎么学

见 `docs/学习步骤.md`。教练小步给代码，你复制 / 读 / 跑。

## 怎么跑（一步一入口）

```text
examples/
  step01_llm/main.py
  step02_adapter/main.py
  step03_stream/main.py
  step04_prompt/main.py
  step05_structured/main.py
src/scenic_cs_agent/   # 可复用库（Adapter 等）
```

当前步骤：

```bash
uv run python examples/step05_structured/main.py
```

