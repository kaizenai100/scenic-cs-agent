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
  step05b_langfuse_prompt/main.py
  step07_agent_loop/main.py
  step08_llm_gateway/main.py
src/scenic_cs_agent/   # 可复用库（Adapter、Gateway、Agent Loop）
```

调模型走进程内 Gateway（`build_gateway_from_config()`）。业务代码拿别名，不自己保管上游 Key，也不在调用点写重试和熔断。说明见 `examples/step08_llm_gateway/README.md`。

当前步骤：

```bash
uv run pytest tests/test_llm_gateway.py -q
uv run python examples/step08_llm_gateway/main.py
```

