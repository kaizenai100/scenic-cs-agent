"""对浏览器 / 客户端的 Run 级 HTTP 入口。

模型适配器吐出的是 ModelStreamEvent，只描述一次模型调用。
这里再包成带序号的 RunEvent，放进内存 EventStore，用 SSE 推出去。
供应商的原始 chunk 不经过这层。
"""

from scenic_cs_agent.api.app import create_app

__all__ = ["create_app"]
