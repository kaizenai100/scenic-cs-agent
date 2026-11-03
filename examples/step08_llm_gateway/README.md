# 步骤 8 · 进程内 LLM Gateway

业务、Agent Loop、示例要调模型时，只拿 `LLMGateway`。

```python
gateway = build_gateway_from_config()
result = gateway.complete(request)
```

不要在业务代码里：

- 读取上游 API Key，再自己创建厂商客户端
- 手写重试、熔断、限流，或按厂商名切换模型

Gateway 负责别名路由、重试、熔断、进程内限流、调用记录、流式取消，以及可选的结构化输出修复。厂商差异仍留在各自的 Adapter。

这不是一个单独部署的 HTTP 服务，没有另一套进程要启动。

```bash
uv run pytest tests/test_llm_gateway.py -q
uv run python examples/step08_llm_gateway/main.py
```

没配密钥时示例走 Fake Adapter。设置了 `DEEPSEEK_API_KEY` 并且 `LIVE=1` 时，才会按 yaml 真调一次。
