"""进程入口。

三种启动方式是同一个 app：

    uv run fastapi dev src/scenic_cs_agent/main.py
    uv run uvicorn scenic_cs_agent.main:app --host 127.0.0.1 --port 8000
    uv run scenic-cs-agent
"""

from __future__ import annotations

import uvicorn

from scenic_cs_agent.bootstrap import create_app

# uvicorn / fastapi dev 导入的就是这个对象。装配发生在 lifespan，不在导入时。
app = create_app()


def main() -> None:
    """脚本入口。监听本机 8000 端口。"""

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
