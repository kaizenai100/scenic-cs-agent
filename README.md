# scenic-cs-agent

景区售前客服 Agent。进程只有一个 HTTP 入口：FastAPI。启动时把配置、Prompt、Gateway、MCP 和工具装进同一个进程，`/v1/runs` 走 Agent Loop。

旧的分步示例留在 tag `learning-steps-v1`，用 `git checkout learning-steps-v1` 查看。

## 启动

默认 `APP_ENV=test`。测试配置的 default 模型是进程内假模型，不需要 API Key，也能完成一轮对话。

```bash
uv sync
uv run fastapi dev src/scenic_cs_agent/main.py
uv run uvicorn scenic_cs_agent.main:app --host 127.0.0.1 --port 8000
uv run scenic-cs-agent
```

生产环境：

```bash
APP_ENV=production uv run uvicorn scenic_cs_agent.main:app --host 127.0.0.1 --port 8000
```

生产配置不拉起仓库里的假 MCP 进程。要接真实模型时，在环境变量里提供 `DEEPSEEK_API_KEY`。密钥只写在环境变量或 `.env`，yaml 里是 `${VAR}`。

## 配置

`config/project.yaml` 是公共默认，`config/project_<env>.yaml` 只写差异，启动时深合并。`APP_ENV` 只接受 `test` / `production`（`dev` 视为 test，`prod` 视为 production）。写错直接启动失败。

| 环境 | 模型 | MCP |
| --- | --- | --- |
| test | `fake` provider，不访问网络 | 拉起 `python -m scenic_cs_agent.mcp.server` |
| production | DeepSeek，密钥来自环境变量 | 默认关闭 |

Langfuse 只在 `gateway.observe.langfuse_enabled` 为真且公钥、私钥都有时才连接。缺密钥时 prompt 用 `src/scenic_cs_agent/prompts/` 下的本地文件，调用不上报。

## 启动顺序

1. 读配置。不合法就失败，不再往下装。
2. 配日志。
3. 按开关连接 Langfuse。
4. 建立 Prompt 注册表，并渲染一次售前模板。
5. 建立一个进程内 Gateway。
6. 连接 MCP。`required: true` 的服务失败则整个进程启动失败。
7. 本地工具和 MCP 工具放进同一个 ToolRuntime。
8. 建立 RunStore 和 AgentService。

关闭时反过来：取消还在跑的 Run，关掉 MCP（stdio 子进程退出），再 flush Langfuse。

## 接口

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| GET | `/healthz` | 进程还在。不检查模型和 MCP。 |
| GET | `/readyz` | 每个 Gateway provider 的熔断状态（`closed` / `open` / `half_open`），以及每台 MCP 的 `connected` / `disabled` / `error` 和 `errors`。 |
| POST | `/v1/runs` | 创建一次对话，202，正文只有 `run_id`。 |
| GET | `/v1/runs/{run_id}/events` | SSE。支持 `Last-Event-ID` 续传。断开连接不会取消 Run。 |
| POST | `/v1/runs/{run_id}/cancel` | 唯一会停掉生成的入口。 |

创建正文：

```json
{"message": "云栖山几点开门", "scene": "presale"}
```

`scene` 默认 `presale`，决定这轮能看见哪些工具。`system` 可选，传了就不用售前模板。

SSE 事件包括 `run.started`、`tool.started`、`tool.completed`、`text.delta`、`message.completed`，以及 `run.completed` / `run.failed` / `run.cancelled`。

本地试一轮：

```bash
curl -s -X POST http://127.0.0.1:8000/v1/runs \
  -H 'content-type: application/json' \
  -d '{"message":"云栖山几点开门"}'
curl -N http://127.0.0.1:8000/v1/runs/<run_id>/events
```

## 目录

```text
src/scenic_cs_agent/
  main.py              # uvicorn / fastapi dev 用的 app
  bootstrap.py         # lifespan：装配和反向释放
  agent/service.py     # 一次对话：Prompt + Gateway + Agent Loop + ToolRuntime
  api/
    deps.py            # 从 app.state 取依赖
    schemas.py
    routes/runs.py
    routes/health.py
    streaming.py       # 仅测试注入的流式路径
  mcp/client.py        # McpHub
  mcp/server.py        # 假票务 MCP（stdio）
  gateway/  runtime/  tools/  prompts/  config/
config/                # project.yaml + 环境覆盖
scripts/sync_prompt.py # 把本地 prompt 推到 Langfuse
tests/
```

## 测试

```bash
uv run pytest -q
```
