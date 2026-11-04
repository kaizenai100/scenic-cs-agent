"""测试用的脚本模型。

按预先写好的列表吐 ModelStreamEvent，或者在指定位置抛异常。
不读密钥，也不访问网络。生产对话不使用它。
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence

from scenic_cs_agent.runtime.model_types import ModelRequest, ModelStreamEvent


class ScriptedModel:
    """按列表播放。items 里混进异常实例时，迭代到那一项就 raise。

    delay 是每一项之前的等待，给取消和断连留出时间窗口。
    用 time.sleep：生产者把 stream() 放在线程里，不会卡住事件循环。
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
        """忽略 request 的正文，按脚本播放。参数留着是为了和适配器同一个形状。"""

        del request
        for item in self._items:
            if self._delay > 0:
                time.sleep(self._delay)
            if isinstance(item, BaseException):
                raise item
            yield item
