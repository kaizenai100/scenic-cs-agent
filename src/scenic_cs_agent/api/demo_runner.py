"""离线脚本模型，以及可选的真实适配器开关。

测试和示例默认走 ScriptedModel：不读密钥、不访问网络。
只有环境变量 SCENIC_SSE_USE_DEEPSEEK=1 时才构造 DeepSeekChatAdapter，
并且密钥从 get_config() 来，不写进代码。

脚本模型实现的是 ModelStreamSource.stream：同步生成器。
生产者会把它放进线程，所以这里用 time.sleep 模拟逐字间隔，不会卡住事件循环。
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator, Sequence

from scenic_cs_agent.api.producer import ModelStreamSource
from scenic_cs_agent.runtime.model_types import ModelRequest, ModelStreamEvent

# 示例用的固定回复。不编造具体票价，和售前助手的边界一致。
DEMO_REPLY = "苏州可以先看拙政园。具体票价我需要先查景区资料，你打算哪一天去？"

DEFAULT_SYSTEM = (
    "你是景区门票售前助手。"
    "不能编造票价或开放时间；没查到资料就说明需要先查，并追问景区名和日期。"
)


class ScriptedModel:
    """按预先写好的列表吐 ModelStreamEvent，或者在指定位置抛异常。

    items 里混进异常实例时，迭代到那一项就 raise。
    用来覆盖「第一条正文之后失败」和「第一条之前失败」而不打真实模型。
    delay 是每一项之前的等待，给取消和断连留出时间窗口。
    """

    def __init__(
        self,
        items: Sequence[ModelStreamEvent | BaseException],
        *,
        delay: float = 0.0,
    ) -> None:
        self._items = list(items)
        self._delay = delay

    def stream(self, request: ModelRequest) -> Iterator[ModelStreamEvent]:
        """忽略 request 的正文，按脚本播放。request 参数留着是为了和适配器同一个形状。"""

        del request
        for item in self._items:
            if self._delay > 0:
                time.sleep(self._delay)
            if isinstance(item, BaseException):
                raise item
            yield item


def demo_items(text: str = DEMO_REPLY) -> list[ModelStreamEvent]:
    """把一段演示回复切成若干 text.delta，并带上正常的结束与用量。

    切块而不是逐字，是为了示例输出可读。结束原因用 stop，走成功路径。
    """

    items: list[ModelStreamEvent] = []
    step = 4
    for index in range(0, len(text), step):
        items.append(ModelStreamEvent(type="text.delta", data={"delta": text[index : index + step]}))
    items.append(ModelStreamEvent(type="model.finished", data={"finish_reason": "stop"}))
    items.append(
        ModelStreamEvent(
            type="model.usage",
            data={
                "input_tokens": 32,
                "output_tokens": len(text),
                "total_tokens": 32 + len(text),
            },
        )
    )
    return items


def build_demo_source() -> ModelStreamSource:
    """按环境变量选择脚本模型或真实 Chat 适配器。

    真实适配器在函数内部才导入：离线测试导入本模块时不应该创建客户端。
    """

    if os.getenv("SCENIC_SSE_USE_DEEPSEEK", "").strip() == "1":
        return _build_deepseek_source()
    # 略作间隔，示例里第一路连接才有机会在中途断开，第二路还能看到后续 seq。
    return ScriptedModel(demo_items(), delay=0.05)


def _build_deepseek_source() -> ModelStreamSource:
    """用现有配置构造适配器。没配密钥就立刻失败，不去发一个必然 401 的请求。"""

    from scenic_cs_agent.config import get_config
    from scenic_cs_agent.runtime import DeepSeekChatAdapter

    config = get_config()
    if not config.model.api_key:
        raise RuntimeError("已设置 SCENIC_SSE_USE_DEEPSEEK=1，但没有 DEEPSEEK_API_KEY")
    return DeepSeekChatAdapter(
        api_key=config.model.api_key,
        model=config.model.name,
        base_url=config.model.base_url,
    )
