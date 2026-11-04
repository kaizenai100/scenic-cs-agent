"""进程启停：配置不合法或必需 MCP 失败时起不来；关掉之后 stdio 子进程要退出。"""

from __future__ import annotations

import sys
import time

import httpx
import pytest
from httpx import ASGITransport

from scenic_cs_agent.bootstrap import create_app
from scenic_cs_agent.config import AppConfig, ConfigError, load_config
from scenic_cs_agent.mcp.client import McpStartupError


def _with_scenic(cfg: AppConfig, **updates: object) -> AppConfig:
    scenic = cfg.mcp.servers["scenic_ticket"].model_copy(update=updates)
    servers = dict(cfg.mcp.servers)
    servers["scenic_ticket"] = scenic
    return cfg.model_copy(update={"mcp": cfg.mcp.model_copy(update={"servers": servers})})


def _alive(pid: int) -> bool:
    try:
        import os

        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _unwrap(exc: BaseException) -> BaseException:
    """Starlette 有时把启动异常包进 ExceptionGroup。测试认最里面那一层。"""

    seen = getattr(exc, "exceptions", None)
    if isinstance(seen, tuple) and seen:
        return _unwrap(seen[0])
    return exc


async def test_illegal_env_fails_before_listen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "staging")
    app = create_app()
    with pytest.raises(ConfigError):
        async with app.router.lifespan_context(app):
            pass


async def test_required_mcp_failure_aborts_startup() -> None:
    cfg = _with_scenic(
        load_config("test"),
        command=sys.executable,
        args=["-c", "import sys; sys.exit(3)"],
        env={},
        startup_timeout_seconds=8,
    )
    app = create_app(config=cfg)
    with pytest.raises(BaseException) as caught:
        async with app.router.lifespan_context(app):
            pass
    assert isinstance(_unwrap(caught.value), McpStartupError)


async def test_lifespan_reports_ready_and_stdio_process_exits(tmp_path) -> None:
    pidfile = tmp_path / "mcp.pid"
    cfg = _with_scenic(
        load_config("test"),
        command=sys.executable,
        args=["-m", "scenic_cs_agent.mcp.server"],
        env={"SCENIC_MCP_PIDFILE": str(pidfile)},
        startup_timeout_seconds=20,
    )
    app = create_app(config=cfg)
    async with app.router.lifespan_context(app):
        assert pidfile.exists()
        pid = int(pidfile.read_text(encoding="utf-8"))
        assert _alive(pid)
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=5) as client:
            health = await client.get("/healthz")
            ready = await client.get("/readyz")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}
        assert ready.status_code == 200
        body = ready.json()
        assert body["status"] == "ok"
        assert body["gateway"]["fake"] == "closed"
        assert body["gateway"]["deepseek"] == "closed"
        assert body["mcp"]["scenic_ticket"]["state"] == "connected"
        assert body["mcp"]["scenic_ticket"]["error"] == ""
        assert body["mcp"]["ticket_http"]["state"] == "disabled"
        assert body["errors"] == []

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and _alive(pid):
        time.sleep(0.05)
    assert not _alive(pid)
