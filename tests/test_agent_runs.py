"""假模型走 POST /v1/runs：工具调用写进 SSE，最后是终止事件。"""

from __future__ import annotations

import json
import sys

import httpx
from httpx import ASGITransport

from scenic_cs_agent.bootstrap import create_app
from scenic_cs_agent.config import AppConfig, load_config


def _with_scenic(cfg: AppConfig, **updates: object) -> AppConfig:
    scenic = cfg.mcp.servers["scenic_ticket"].model_copy(update=updates)
    servers = dict(cfg.mcp.servers)
    servers["scenic_ticket"] = scenic
    return cfg.model_copy(update={"mcp": cfg.mcp.model_copy(update={"servers": servers})})


def parse_sse(raw: str) -> list[dict]:
    events: list[dict] = []
    for block in raw.split("\n\n"):
        if block == "" or block.startswith(":"):
            continue
        data_line = next(line for line in block.split("\n") if line.startswith("data: "))
        events.append(json.loads(data_line.removeprefix("data: ")))
    return events


async def test_post_run_calls_tool_then_completes() -> None:
    cfg = _with_scenic(
        load_config("test"),
        command=sys.executable,
        args=["-m", "scenic_cs_agent.mcp.server"],
        startup_timeout_seconds=20,
    )
    app = create_app(config=cfg)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=10) as client:
            created = await client.post("/v1/runs", json={"message": "云栖山几点开门"})
            assert created.status_code == 202, created.text
            run_id = created.json()["run_id"]
            assert run_id.startswith("run_")
            async with client.stream("GET", f"/v1/runs/{run_id}/events") as response:
                assert response.status_code == 200
                assert response.headers["content-type"].startswith("text/event-stream")
                raw = (await response.aread()).decode()

    events = parse_sse(raw)
    kinds = [event["type"] for event in events]
    assert kinds[0] == "run.started"
    assert "tool.started" in kinds
    assert "tool.completed" in kinds
    assert "text.delta" in kinds
    assert kinds[-2] == "message.completed"
    assert kinds[-1] == "run.completed"
    started = next(event for event in events if event["type"] == "tool.started")
    finished = next(event for event in events if event["type"] == "tool.completed")
    assert started["data"] == {"tool_call_id": "call_fake_open_hours", "name": "get_open_hours"}
    assert finished["data"]["name"] == "get_open_hours"
    assert finished["data"]["ok"] is True
    assert finished["data"]["tool_call_id"] == started["data"]["tool_call_id"]
    deltas = "".join(event["data"]["delta"] for event in events if event["type"] == "text.delta")
    assert deltas == next(event["data"]["text"] for event in events if event["type"] == "message.completed")
    assert "不会另外编造" in deltas
    assert events[-1]["data"]["finish_reason"] == "stop"
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
