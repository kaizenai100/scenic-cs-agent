"""步骤 8：进程内 LLM Gateway。

业务和 Agent Loop 只通过 Gateway 调模型：
  - 不在业务代码里读取上游 API Key，也不自己 new 厂商客户端
  - 不在业务代码里写重试、熔断、限流、换厂商
  - 要换模型，改配置里的别名路由（default / fast → provider + model）

适配器仍然只负责某一家的协议。Gateway 负责路由、重试、熔断、限流、记录和取消。
这不是一个要单独部署的 HTTP 服务。

本文件默认全程 Fake，不需要任何 Key。
LIVE=1 且配置了 DEEPSEEK_API_KEY 时，末尾会按 yaml 真调一次。

运行：uv run python examples/step08_llm_gateway/main.py
"""

from __future__ import annotations

import json
import os
import threading

from dotenv import load_dotenv

from scenic_cs_agent.config import load_config
from scenic_cs_agent.gateway import (
    AliasSpec,
    AllRoutesFailed,
    CircuitOpenError,
    GatewayError,
    LLMGateway,
    PartialStreamError,
    ProviderBinding,
    RateLimitExceeded,
    RetryPolicy,
    RouteSpec,
    StreamCancelled,
    UpstreamError,
    build_gateway_from_config,
)
from scenic_cs_agent.runtime.agent_loop import run_agent_loop
from scenic_cs_agent.runtime.loop_guard import LoopGuard
from scenic_cs_agent.runtime.model_types import (
    ModelAdapter,
    ModelCapabilities,
    ModelRequest,
    ModelResult,
    ModelStreamEvent,
)
from scenic_cs_agent.tools import SCENIC_TOOLS, ExecutionContext, ToolRuntime

load_dotenv()


class DemoClock:
    """示例用的假时钟，重试和熔断不用真的睡觉。"""

    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self.t += seconds

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeAdapter(ModelAdapter):
    """脚本适配器：按队列返回结果或抛上游错误。"""

    name = "fake"
    capabilities = ModelCapabilities(
        chat_completions=True,
        responses=False,
        structured_output="prompt_only",
        tool_calling=True,
        supports_temperature=True,
        supports_top_p=True,
    )

    def __init__(self, script: list[ModelResult | Exception] | None = None, *, text: str = "ok") -> None:
        self.calls = 0
        self._script = list(script or [])
        self._text = text

    def generate(self, request: ModelRequest) -> ModelResult:
        del request
        self.calls += 1
        if self._script:
            item = self._script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return ModelResult(kind="text", text=self._text, finish_reason="stop", input_tokens=1, output_tokens=1)


def _banner(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def _request(user: str = "拙政园明天开门吗？") -> ModelRequest:
    return ModelRequest(system="你是景区售前客服。不要编造票价。", user=user)


def _ok(text: str) -> ModelResult:
    return ModelResult(kind="text", text=text, finish_reason="stop", input_tokens=8, output_tokens=len(text))


def _gateway(
    adapters: dict[str, FakeAdapter],
    *,
    max_attempts: int = 2,
    threshold: int = 5,
    cooldown: float = 30,
    burst: int = 10,
    rpm: int = 60,
    rate_limit_enabled: bool = True,
    base_delay: float = 0.0,
    max_delay: float = 0.0,
    clock: DemoClock | None = None,
) -> tuple[LLMGateway, DemoClock]:
    clock = clock or DemoClock()
    bindings = {
        name: ProviderBinding(name=name, adapter=adapter, model=f"{name}-model")
        for name, adapter in adapters.items()
    }
    gateway = LLMGateway(
        bindings=bindings,
        aliases={
            "default": AliasSpec(
                strategy="priority",
                routes=tuple(
                    RouteSpec(provider=name, model=f"{name}-model") for name in adapters
                ),
            )
        },
        default_alias="default",
        retry=RetryPolicy(
            max_attempts_per_route=max_attempts,
            base_delay_seconds=base_delay,
            max_delay_seconds=max_delay,
        ),
        clock=clock,
        failure_threshold=threshold,
        cooldown_seconds=cooldown,
        rate_limit_enabled=rate_limit_enabled,
        requests_per_minute=rpm,
        burst=burst,
    )
    return gateway, clock


def _show_record(gateway: LLMGateway) -> None:
    record = gateway.records[-1]
    print(
        "record:",
        json.dumps(
            {
                "status": record.status,
                "alias": record.alias,
                "provider": record.provider,
                "model": record.model,
                "attempts": record.attempts,
                "retries": record.retries,
                "fallback": record.fallback,
                "error_code": record.error_code,
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "latency_ms": round(record.latency_ms, 1),
            },
            ensure_ascii=False,
        ),
    )


def demo_success() -> None:
    _banner("1) 直通成功")
    primary = FakeAdapter([_ok("拙政园正常开放，具体票价需要再查。")])
    backup = FakeAdapter([_ok("不该出现")])
    gateway, _clock = _gateway({"primary": primary, "backup": backup})
    result = gateway.complete(_request())
    print(result.text)
    print(f"primary.calls={primary.calls} backup.calls={backup.calls}")
    _show_record(gateway)


def demo_retry() -> None:
    _banner("2) 429 后同一路由重试成功")
    primary = FakeAdapter(
        [
            UpstreamError("上游 429", code="429", retryable=True),
            _ok("已重试成功，仍不编造票价。"),
        ]
    )
    backup = FakeAdapter([_ok("不该 fallback")])
    gateway, clock = _gateway(
        {"primary": primary, "backup": backup},
        max_attempts=2,
        base_delay=0.2,
        max_delay=1,
    )
    result = gateway.complete(_request())
    print(result.text)
    print(f"primary.calls={primary.calls} slept={clock.t}")
    _show_record(gateway)


def demo_fallback() -> None:
    _banner("3) 主路由失败后 fallback")
    primary = FakeAdapter([UpstreamError("主挂了", code="500", retryable=True)])
    backup = FakeAdapter([_ok("备路由回答：请告诉我具体景区名。")])
    gateway, _clock = _gateway({"primary": primary, "backup": backup}, max_attempts=1)
    result = gateway.complete(_request())
    print(result.text)
    print(f"primary.calls={primary.calls} backup.calls={backup.calls}")
    _show_record(gateway)


def demo_circuit() -> None:
    _banner("4) 熔断打开后拒绝")
    primary = FakeAdapter()
    primary.generate = _always_fail(primary)  # type: ignore[method-assign]
    gateway, clock = _gateway({"primary": primary}, max_attempts=1, threshold=2, cooldown=10)
    for _ in range(2):
        try:
            gateway.complete(_request())
        except AllRoutesFailed as exc:
            print(f"上游失败 code={exc.code} calls={primary.calls} state={gateway.breaker_state('primary')}")
    try:
        gateway.complete(_request())
    except CircuitOpenError as exc:
        print(f"熔断拒绝 provider={exc.provider} calls 仍为 {primary.calls}")
    clock.advance(10)
    print(f"冷却后 state={gateway.breaker_state('primary')}（半开，下一次才是探测）")


def _always_fail(adapter: FakeAdapter):
    def generate(request: ModelRequest) -> ModelResult:
        del request
        adapter.calls += 1
        raise UpstreamError("持续失败", code="500", retryable=True)

    return generate


def demo_rate_limit() -> None:
    _banner("5) 进程内限流")
    primary = FakeAdapter([_ok("第一次通过")])
    gateway, clock = _gateway({"primary": primary}, burst=1, rpm=60)
    print(gateway.complete(_request()).text)
    try:
        gateway.complete(_request())
    except RateLimitExceeded as exc:
        print(f"第二次被限流 code={exc.code} calls={primary.calls}")
    clock.advance(1)
    primary._script.append(_ok("补桶之后又通过"))
    print(gateway.complete(_request()).text)


def demo_stream_cancel() -> None:
    _banner("6) 流式取消")

    class CancelAdapter(FakeAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False

        def stream(self, request: ModelRequest, *, cancel_event: threading.Event | None = None):
            del request
            self.calls += 1
            try:
                yield ModelStreamEvent(type="text.delta", data={"delta": "拙政园"})
                while cancel_event is None or not cancel_event.is_set():
                    threading.Event().wait(0.01)
            finally:
                self.closed = True

    adapter = CancelAdapter()
    gateway, _clock = _gateway({"primary": adapter})
    cancel = threading.Event()
    parts: list[str] = []
    try:
        for event in gateway.stream(_request(), cancel_event=cancel):
            if event.type == "text.delta":
                parts.append(str(event.data["delta"]))
                cancel.set()
    except StreamCancelled:
        print(f"已取消 text={''.join(parts)} upstream_closed={adapter.closed}")
    _show_record(gateway)


def demo_no_resume_after_text() -> None:
    _banner("7) 已经吐出正文就不能换路由续写")

    class PartialAdapter(FakeAdapter):
        def stream(self, request: ModelRequest, *, cancel_event: threading.Event | None = None):
            del request, cancel_event
            self.calls += 1
            yield ModelStreamEvent(type="text.delta", data={"delta": "半段"})
            raise UpstreamError("流中断", code="500", retryable=True)

    primary = PartialAdapter()
    backup = FakeAdapter()
    backup.calls = 0

    def backup_stream(request: ModelRequest, *, cancel_event: threading.Event | None = None):
        del request, cancel_event
        backup.calls += 1
        yield ModelStreamEvent(type="text.delta", data={"delta": "不该接上"})

    backup.stream = backup_stream  # type: ignore[method-assign]
    gateway, _clock = _gateway({"primary": primary, "backup": backup}, max_attempts=2)
    stream = gateway.stream(_request())
    print("first:", next(stream).data["delta"])
    try:
        next(stream)
    except PartialStreamError as exc:
        print(f"停止续写 code={exc.code} backup.calls={backup.calls}")


def demo_loop_through_gateway() -> None:
    _banner("8) Agent Loop 把 Gateway 当适配器")
    call = {
        "id": "call_hours_1",
        "name": "get_open_hours",
        "arguments": json.dumps({"scenic_name": "拙政园"}, ensure_ascii=False),
    }
    adapter = FakeAdapter(
        [
            ModelResult(kind="tool_calls", tool_calls=[call], text=""),
            _ok("拙政园明天按园内公告时间开放。票价我需要再查，这里不编。"),
        ]
    )
    gateway, _clock = _gateway({"primary": adapter}, max_attempts=1, burst=10)
    result = run_agent_loop(
        adapter=gateway,
        runtime=ToolRuntime(SCENIC_TOOLS),
        ctx=ExecutionContext(run_id="step08", user_id="u-demo", channel="app", scene="presale"),
        system="你是景区售前客服。",
        user="拙政园明天几点开门？",
        guard=LoopGuard(max_turns=4),
        on_event=lambda *_args: None,
    )
    print(f"stop_reason={result.stop_reason} gateway_calls={len(gateway.records)}")
    print(result.text)


def demo_factory() -> None:
    _banner("9) build_gateway_from_config（Fake 注入，不读 Key）")
    cfg = load_config("test")
    gateway = build_gateway_from_config(
        cfg,
        adapters={
            "deepseek": FakeAdapter([_ok("配置里的 default 打到 deepseek")]),
            "deepseek_backup": FakeAdapter([_ok("这条备路由这次不会被选中")]),
        },
        clock=DemoClock(),
    )
    print(gateway.complete(_request()).text)
    print(
        "yaml:",
        f"alias={cfg.gateway.default_alias}",
        f"test_rpm={cfg.gateway.rate_limit.requests_per_minute}",
        f"test_threshold={cfg.gateway.circuit_breaker.failure_threshold}",
        f"routes={[route.provider for route in cfg.gateway.aliases['default'].routes]}",
    )
    _show_record(gateway)


def demo_live() -> None:
    _banner("10) 真模型（可选）")
    if os.getenv("LIVE") != "1":
        print("未设置 LIVE=1，跳过真调。前面的 Fake 段不需要 API Key。")
        return
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("LIVE=1 但没有 DEEPSEEK_API_KEY，跳过真调。")
        return
    gateway = build_gateway_from_config(load_config())
    try:
        result = gateway.complete(
            ModelRequest(
                system="你是景区售前客服。不能编造票价或开放时间；没查资料就说明需要先查。",
                user="想去苏州玩，有什么园子？门票多少钱？",
                max_output_tokens=256,
                temperature=0.2,
            )
        )
    except GatewayError as exc:
        print(f"真调失败 code={exc.code} {exc}")
        return
    print(result.text)
    _show_record(gateway)


if __name__ == "__main__":
    demo_success()
    demo_retry()
    demo_fallback()
    demo_circuit()
    demo_rate_limit()
    demo_stream_cancel()
    demo_no_resume_after_text()
    demo_loop_through_gateway()
    demo_factory()
    demo_live()
