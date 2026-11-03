# 步骤 8 · 进程内 LLM Gateway

业务、Agent Loop、示例要调模型时，只拿 `LLMGateway`。

```python
gateway = build_gateway_from_config()
result = gateway.complete(request)
```

不要在业务代码里：

- 读取上游 API Key，再自己创建厂商客户端
- 手写重试、熔断、限流，或按厂商名切换模型

Gateway 负责别名路由（priority 或 weighted_round_robin）、重试、熔断、进程内限流、调用记录、流式取消，以及可选的结构化输出修复。厂商差异仍留在各自的 Adapter。

流式调用的 `CallRecord.ttft_ms` 是交出第一条事件的耗时（毫秒）。非流式这个字段是 `None`。

观测默认接到现有 Langfuse 配置（`langfuse.host` / 密钥 / `environment`），不另建 prompt。`gateway.observe.langfuse_enabled=false`，或者没配 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` 时，什么都不上报，模型调用照常进行。上报内容只有别名、provider、model、耗时、token、重试、fallback、ttft、错误码，不含对话正文。

`aliases.fast` 是加权轮询示例：`weight` 越大，这一轮越先被选中；选中的那条失败后，仍按其余路由 fallback。`default` 保持 `priority`，按列表顺序试。

这不是一个单独部署的 HTTP 服务，没有另一套进程要启动。

```bash
uv run pytest tests/test_llm_gateway.py -q
uv run python examples/step08_llm_gateway/main.py
```

没配密钥时示例走 Fake Adapter。设置了 `DEEPSEEK_API_KEY` 并且 `LIVE=1` 时，才会按 yaml 真调一次。
