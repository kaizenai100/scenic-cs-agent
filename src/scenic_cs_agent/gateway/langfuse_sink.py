"""把一次 Gateway 调用记成 Langfuse 的 generation。

只复用配置里已有的 Langfuse 连接（host、密钥、environment）。
这里不拉 prompt、不创建 prompt 版本。prompt 仍由 prompts/registry.py 负责，避免两套提示词系统。

记什么：别名、provider、model、耗时、token、重试、是否 fallback、ttft、错误码。
不记用户原文和模型正文。调用记录默认会离开本进程，正文要不要送出得单独决定，不在观测开关里偷偷带上。

Langfuse 写失败只打日志。模型调用该成功还是成功，该失败还是原来的 GatewayError。
没配密钥，或 observe.langfuse_enabled=false 时，工厂不会挂这个回调，等于什么都不做。
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from scenic_cs_agent.gateway.observe import CallRecord

logger = logging.getLogger("scenic_cs_agent.gateway")


class _Observation(Protocol):
    def end(self) -> object:
        """结束这一条 generation。"""


class LangfuseObservationClient(Protocol):
    """只要有 start_observation 就能当观测客户端。测试用假对象，不必连真的 Langfuse。"""

    def start_observation(self, **kwargs: Any) -> _Observation:
        """打开一条 generation，返回带 end() 的对象。"""


class LangfuseCallSink:
    """on_record 回调：一条 CallRecord 对应一条 generation。"""

    def __init__(self, client: LangfuseObservationClient) -> None:
        self._client = client

    def __call__(self, record: CallRecord) -> None:
        try:
            self._emit(record)
        except Exception:
            logger.exception("Langfuse 观测写入失败，不影响本次模型调用")

    def _emit(self, record: CallRecord) -> None:
        # 流式和非流式分开命名，看板里能直接看出这条 trace 有没有 TTFT。
        name = "gateway.stream" if record.stream else "gateway.complete"
        observation = self._client.start_observation(
            name=name,
            as_type="generation",
            model=record.model,
            # 不传 input/output。正文不在 CallRecord 里，这里也不另去捞。
            metadata={
                "call_id": record.call_id,
                "alias": record.alias,
                "provider": record.provider,
                "model": record.model,
                "status": record.status,
                "latency_ms": record.latency_ms,
                "ttft_ms": record.ttft_ms,
                "retries": record.retries,
                "attempts": record.attempts,
                "fallback": record.fallback,
                "error_code": record.error_code,
                "repair_count": record.repair_count,
            },
            usage_details={
                "input": record.input_tokens,
                "output": record.output_tokens,
            },
            level=_level(record),
            status_message=record.error_code,
        )
        observation.end()


def _level(record: CallRecord) -> str:
    """Langfuse 的 level。成功是 DEFAULT；限流和取消是警告；上游或熔断失败是错误。"""
    if record.status in {"error", "circuit_open"}:
        return "ERROR"
    if record.status in {"cancelled", "rate_limited"}:
        return "WARNING"
    return "DEFAULT"


def langfuse_sink_from_config(
    *,
    enabled: bool,
    host: str,
    public_key: str | None,
    secret_key: str | None,
    environment: str,
    client: LangfuseObservationClient | None = None,
) -> LangfuseCallSink | None:
    """按配置决定要不要上报。

    显式传入 client 时只在 enabled 时使用，测试靠这个注入假客户端。
    没传 client 时，两边密钥都有才创建真客户端；缺一个就返回 None，不连网。
    """
    if not enabled:
        return None
    if client is not None:
        return LangfuseCallSink(client)
    if not public_key or not secret_key:
        logger.info("Langfuse 观测跳过：public_key 或 secret_key 未配置")
        return None
    # 延迟导入：没启用观测时，导入 Gateway 不必去初始化 Langfuse 的导出器。
    from langfuse import Langfuse

    real = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
        environment=environment,
    )
    return LangfuseCallSink(real)
