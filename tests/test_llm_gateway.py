"""步骤 8 · 进程内 LLM Gateway。

全部用 Fake Adapter，不读 API Key，也不访问网络。
覆盖：直通成功、失败后重试成功、熔断打开拒绝、限流、fallback、流式取消、
加权轮询、流式 TTFT、Langfuse 观测开关（假客户端，不连真的 Langfuse）。
另外锁住几件骨架里容易写错的行为：半开探测、正文已输出不能换路由、配置深合并、结构化修复。
"""

from __future__ import annotations

import json
import threading

import pytest
from pydantic import ValidationError

from scenic_cs_agent.config import ConfigError, GatewayConfig, load_config
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
    StructuredOutputError,
    UpstreamError,
    build_gateway_from_config,
)
from scenic_cs_agent.gateway.classify import classify_exception
from scenic_cs_agent.gateway.clock import Clock
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


class FakeClock:
    """测试钟：sleep 只把时间往前拨，不阻塞。"""

    def __init__(self) -> None:
        self.t = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        if seconds > 0:
            self.t += seconds

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeAdapter(ModelAdapter):
    """按脚本吐结果或抛异常。脚本耗尽后返回固定文本。"""

    name = "fake"
    capabilities = ModelCapabilities(
        chat_completions=True,
        responses=False,
        structured_output="json_mode",
        tool_calling=True,
        supports_temperature=True,
        supports_top_p=True,
    )

    def __init__(
        self,
        script: list[ModelResult | Exception] | None = None,
        *,
        text: str = "ok",
    ) -> None:
        self.calls = 0
        self.requests: list[ModelRequest] = []
        self._script = list(script or [])
        self._text = text

    def generate(self, request: ModelRequest) -> ModelResult:
        self.calls += 1
        self.requests.append(request)
        if self._script:
            item = self._script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return _text_result(self._text)


def _text_result(text: str, *, input_tokens: int = 1, output_tokens: int = 1) -> ModelResult:
    return ModelResult(
        kind="text",
        text=text,
        finish_reason="stop",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _request() -> ModelRequest:
    return ModelRequest(system="你是景区售前客服。", user="拙政园明天开门吗？")


def _build(
    adapters: dict[str, FakeAdapter],
    *,
    aliases: dict[str, list[str]] | None = None,
    default_alias: str = "default",
    max_attempts: int = 2,
    base_delay: float = 0.0,
    max_delay: float = 0.0,
    threshold: int = 5,
    cooldown: float = 30.0,
    burst: int = 10,
    rpm: int = 600,
    rate_limit_enabled: bool = True,
    structured_retries: int = 1,
    clock: Clock | None = None,
    on_record: object | None = None,
    strategy: str = "priority",
    weights: dict[str, int] | None = None,
) -> tuple[LLMGateway, FakeClock]:
    """用假适配器组一个网关。aliases 缺省时，default 按 adapters 的插入顺序做 priority。"""
    fake_clock = clock if isinstance(clock, FakeClock) else FakeClock()
    used = fake_clock if clock is None else clock
    if not isinstance(used, FakeClock):
        raise AssertionError("测试只注入 FakeClock")
    bindings = {
        name: ProviderBinding(name=name, adapter=adapter, model=f"{name}-model")
        for name, adapter in adapters.items()
    }
    alias_map = aliases or {"default": list(adapters)}
    weight_of = weights or {}
    specs = {
        alias: AliasSpec(
            strategy=strategy,
            routes=tuple(
                RouteSpec(
                    provider=provider,
                    model=f"{provider}-model",
                    weight=weight_of.get(provider, 1),
                )
                for provider in providers
            ),
        )
        for alias, providers in alias_map.items()
    }
    gateway = LLMGateway(
        bindings=bindings,
        aliases=specs,
        default_alias=default_alias,
        retry=RetryPolicy(
            max_attempts_per_route=max_attempts,
            base_delay_seconds=base_delay,
            max_delay_seconds=max_delay,
        ),
        clock=used,
        failure_threshold=threshold,
        cooldown_seconds=cooldown,
        rate_limit_enabled=rate_limit_enabled,
        requests_per_minute=rpm,
        burst=burst,
        structured_output_retries=structured_retries,
        on_record=on_record,  # type: ignore[arg-type]
    )
    return gateway, used


def test_complete_success_records_route_and_tokens():
    """主路由直接成功：不重试、不 fallback，记录里能看到别名、provider、token。"""
    seen: list[object] = []
    primary = FakeAdapter(text="拙政园正常开放。")
    backup = FakeAdapter(text="不该走到备路由")
    gateway, _clock = _build(
        {"primary": primary, "backup": backup},
        on_record=seen.append,
    )
    primary._text = "拙政园正常开放。"
    # 覆盖默认 token，确认记录抄的是结果而不是写死的数。
    primary._script = [_text_result("拙政园正常开放。", input_tokens=11, output_tokens=7)]

    result = gateway.complete(_request(), alias="default")

    assert result.text == "拙政园正常开放。"
    assert primary.calls == 1
    assert backup.calls == 0
    record = gateway.records[-1]
    assert record.status == "ok"
    assert record.alias == "default"
    assert record.provider == "primary"
    assert record.model == "primary-model"
    assert record.attempts == 1
    assert record.retries == 0
    assert record.fallback is False
    assert record.error_code is None
    assert record.input_tokens == 11
    assert record.output_tokens == 7
    assert record.stream is False
    assert record.ttft_ms is None
    assert seen == [record]


def test_retry_then_success_does_not_fallback():
    """429 可重试：同一路由退避后再成功，备路由不被调用。"""
    primary = FakeAdapter(
        [
            UpstreamError("限流", code="429", retryable=True),
            _text_result("修好了", input_tokens=2, output_tokens=3),
        ]
    )
    backup = FakeAdapter(text="备路由")
    gateway, clock = _build(
        {"primary": primary, "backup": backup},
        max_attempts=2,
        base_delay=0.2,
        max_delay=1.0,
    )

    result = gateway.complete(_request())

    assert result.text == "修好了"
    assert primary.calls == 2
    assert backup.calls == 0
    assert clock.slept == [0.2]
    record = gateway.records[-1]
    assert record.status == "ok"
    assert record.retries == 1
    assert record.attempts == 2
    assert record.fallback is False
    assert record.provider == "primary"
    assert record.input_tokens == 2
    assert record.output_tokens == 3


def test_non_retryable_does_not_spin_then_falls_back():
    """400 不在同一条路由上重试，但可以换备路由。"""
    primary = FakeAdapter([UpstreamError("参数错误", code="400", retryable=False)])
    backup = FakeAdapter(text="备用回答")
    gateway, clock = _build(
        {"primary": primary, "backup": backup},
        max_attempts=3,
        base_delay=0.5,
        max_delay=2.0,
    )

    result = gateway.complete(_request())

    assert result.text == "备用回答"
    assert primary.calls == 1
    assert backup.calls == 1
    assert clock.slept == []
    record = gateway.records[-1]
    assert record.fallback is True
    assert record.provider == "backup"
    assert record.retries == 0
    assert record.attempts == 2


def test_circuit_open_rejects_without_calling_adapter():
    """失败次数到阈值后，下一次调用在打适配器之前被拒绝。"""

    class AlwaysFail(FakeAdapter):
        def generate(self, request: ModelRequest) -> ModelResult:
            self.calls += 1
            raise UpstreamError("挂了", code="500", retryable=True)

    primary = AlwaysFail()
    gateway, _clock = _build(
        {"primary": primary},
        max_attempts=1,
        threshold=2,
        burst=10,
    )

    with pytest.raises(AllRoutesFailed):
        gateway.complete(_request())
    with pytest.raises(AllRoutesFailed):
        gateway.complete(_request())
    assert gateway.breaker_state("primary") == "open"
    assert primary.calls == 2

    with pytest.raises(CircuitOpenError):
        gateway.complete(_request())

    assert primary.calls == 2
    record = gateway.records[-1]
    assert record.status == "circuit_open"
    assert record.error_code == "circuit_open"
    assert record.attempts == 0
    assert record.provider == "primary"


def test_circuit_is_shared_across_aliases():
    """同一 provider 被两个别名使用时，熔断只记一份。否则换个别名就能绕开。"""
    primary = FakeAdapter([UpstreamError("挂了", code="500", retryable=True)])
    gateway, _clock = _build(
        {"primary": primary},
        aliases={"default": ["primary"], "fast": ["primary"]},
        max_attempts=1,
        threshold=1,
    )

    with pytest.raises(AllRoutesFailed):
        gateway.complete(_request(), alias="default")

    with pytest.raises(CircuitOpenError):
        gateway.complete(_request(), alias="fast")
    assert primary.calls == 1


def test_half_open_probe_success_closes_breaker():
    """冷却结束后只放探测；探测成功则关上，之后恢复正常调用。"""
    primary = FakeAdapter(
        [
            UpstreamError("挂了", code="500", retryable=True),
            _text_result("探测成功"),
            _text_result("恢复后"),
        ]
    )
    gateway, clock = _build(
        {"primary": primary},
        max_attempts=1,
        threshold=1,
        cooldown=10,
    )

    with pytest.raises(AllRoutesFailed):
        gateway.complete(_request())
    assert gateway.breaker_state("primary") == "open"

    with pytest.raises(CircuitOpenError):
        gateway.complete(_request())
    assert primary.calls == 1

    clock.advance(10)
    assert gateway.breaker_state("primary") == "half_open"
    assert gateway.complete(_request()).text == "探测成功"
    assert gateway.breaker_state("primary") == "closed"
    assert gateway.complete(_request()).text == "恢复后"
    assert primary.calls == 3


def test_rate_limit_rejects_then_refills():
    """桶空时拒绝且不打上游；时间往前走会按 RPM 补令牌。"""
    primary = FakeAdapter(text="通过")
    gateway, clock = _build(
        {"primary": primary},
        burst=1,
        rpm=60,
        rate_limit_enabled=True,
    )

    assert gateway.complete(_request()).text == "通过"
    with pytest.raises(RateLimitExceeded):
        gateway.complete(_request())
    assert primary.calls == 1
    assert gateway.records[-1].status == "rate_limited"
    assert gateway.records[-1].error_code == "rate_limited"
    assert gateway.records[-1].attempts == 0

    clock.advance(1)  # 60 次/分钟 = 每秒补 1 个
    assert gateway.complete(_request()).text == "通过"
    assert primary.calls == 2


def test_rate_limit_can_be_disabled():
    primary = FakeAdapter(text="通过")
    gateway, _clock = _build({"primary": primary}, burst=1, rate_limit_enabled=False)
    for _ in range(3):
        assert gateway.complete(_request()).text == "通过"
    assert primary.calls == 3


def test_stream_cancel_stops_upstream():
    """调用方取消后，适配器要看到 cancel，并且生成器被关掉。"""

    class CancelAdapter(FakeAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.saw_cancel = False
            self.closed = False

        def stream(self, request: ModelRequest, *, cancel_event: threading.Event | None = None):
            del request
            self.calls += 1
            try:
                yield ModelStreamEvent(type="text.delta", data={"delta": "你"})
                for _ in range(200):
                    if cancel_event is not None and cancel_event.is_set():
                        self.saw_cancel = True
                        return
                    # 给取消留一个很短的窗口；测试会在下一次 next 之前把开关打开。
                    threading.Event().wait(0.01)
                yield ModelStreamEvent(type="text.delta", data={"delta": "不该出现"})
            finally:
                self.closed = True

    adapter = CancelAdapter()
    gateway, _clock = _build({"primary": adapter}, max_attempts=2)
    cancel = threading.Event()
    stream = gateway.stream(_request(), cancel_event=cancel)

    first = next(stream)
    assert first.data["delta"] == "你"
    cancel.set()
    with pytest.raises(StreamCancelled):
        next(stream)

    assert adapter.calls == 1
    assert adapter.saw_cancel is True
    assert adapter.closed is True
    record = gateway.records[-1]
    assert record.status == "cancelled"
    assert record.error_code == "cancelled"
    assert record.stream is True
    assert record.provider == "primary"
    assert record.attempts == 1
    assert record.ttft_ms == 0.0


def test_stream_does_not_fallback_after_text_is_delivered():
    """已经把正文交给调用方之后，备路由不能再续写。"""

    class PartialAdapter(FakeAdapter):
        def stream(self, request: ModelRequest, *, cancel_event: threading.Event | None = None):
            del request, cancel_event
            self.calls += 1
            yield ModelStreamEvent(type="text.delta", data={"delta": "半"})
            raise UpstreamError("中途断开", code="500", retryable=True)

    primary = PartialAdapter()
    backup = FakeAdapter(text="备路由不该被叫")
    backup_calls = {"stream": 0}

    def backup_stream(request: ModelRequest, *, cancel_event: threading.Event | None = None):
        del request, cancel_event
        backup_calls["stream"] += 1
        yield ModelStreamEvent(type="text.delta", data={"delta": "备"})

    backup.stream = backup_stream  # type: ignore[method-assign]
    gateway, _clock = _build(
        {"primary": primary, "backup": backup},
        max_attempts=2,
    )
    stream = gateway.stream(_request())
    assert next(stream).data["delta"] == "半"
    with pytest.raises(PartialStreamError):
        next(stream)

    assert primary.calls == 1
    assert backup_calls["stream"] == 0
    assert gateway.records[-1].status == "error"
    assert gateway.records[-1].error_code == "stream_partial"
    assert gateway.records[-1].fallback is False


def test_stream_fallback_before_any_event():
    """一个事件都还没交出时，主路由失败可以换备路由。"""

    class BoomAdapter(FakeAdapter):
        def stream(self, request: ModelRequest, *, cancel_event: threading.Event | None = None):
            del request, cancel_event
            self.calls += 1
            raise UpstreamError("连不上", code="503", retryable=True)
            yield ModelStreamEvent(type="text.delta", data={"delta": "不会执行"})

    class BackupAdapter(FakeAdapter):
        def stream(self, request: ModelRequest, *, cancel_event: threading.Event | None = None):
            del request, cancel_event
            self.calls += 1
            yield ModelStreamEvent(type="text.delta", data={"delta": "备用"})
            yield ModelStreamEvent(
                type="model.usage",
                data={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
            )

    primary = BoomAdapter()
    backup = BackupAdapter()
    gateway, _clock = _build(
        {"primary": primary, "backup": backup},
        max_attempts=2,
        base_delay=0.0,
        max_delay=0.0,
    )

    events = list(gateway.stream(_request()))
    assert [event.data.get("delta") for event in events if event.type == "text.delta"] == ["备用"]
    assert primary.calls == 2
    assert backup.calls == 1
    record = gateway.records[-1]
    assert record.status == "ok"
    assert record.fallback is True
    assert record.provider == "backup"
    assert record.retries == 1
    assert record.input_tokens == 4
    assert record.output_tokens == 2


def test_structured_repair_feeds_error_back_once():
    """校验失败把错误附到下一次请求；修复成功后不再打第三次。"""
    adapter = FakeAdapter(
        [
            _text_result("not-json"),
            _text_result('{"ok": true}', input_tokens=5, output_tokens=6),
        ]
    )
    gateway, _clock = _build({"primary": adapter}, structured_retries=1)
    original = ModelRequest(
        system="只输出 JSON",
        user="识别意图",
        messages=[
            {"role": "system", "content": "只输出 JSON"},
            {"role": "user", "content": "识别意图"},
        ],
        output_schema={"type": "object"},
    )

    def validate(result: ModelResult) -> str | None:
        try:
            json.loads(result.text or "")
        except json.JSONDecodeError as exc:
            return str(exc)
        return None

    result = gateway.complete(original, output_validator=validate)

    assert result.text == '{"ok": true}'
    assert adapter.calls == 2
    assert original.messages is not None
    assert len(original.messages) == 2  # 调用方手里的历史不被网关改掉
    repair_messages = adapter.requests[1].messages or []
    assert repair_messages[-1]["role"] == "user"
    assert "校验错误" in repair_messages[-1]["content"]
    assert "not-json" in repair_messages[-1]["content"]
    record = gateway.records[-1]
    assert record.repair_count == 1
    assert record.retries == 0
    assert record.status == "ok"
    # 两次成功返回的 token 都算进这一次业务调用。
    assert record.input_tokens == 6
    assert record.output_tokens == 7


def test_structured_repair_stops_when_budget_is_zero():
    adapter = FakeAdapter([_text_result("not-json"), _text_result('{"ok": true}')])
    gateway, _clock = _build({"primary": adapter}, structured_retries=0)

    def validate(result: ModelResult) -> str | None:
        try:
            json.loads(result.text or "")
        except json.JSONDecodeError as exc:
            return str(exc)
        return None

    with pytest.raises(StructuredOutputError) as caught:
        gateway.complete(_request(), output_validator=validate, max_repairs=0)
    assert caught.value.code == "structured_invalid"
    assert caught.value.result.text == "not-json"
    assert adapter.calls == 1


def test_agent_loop_uses_gateway_as_adapter():
    """Loop 不改签名：Gateway 实现 generate，工具 id 仍能贯穿回灌。"""
    call = {
        "id": "call_hours_1",
        "name": "get_open_hours",
        "arguments": json.dumps({"scenic_name": "拙政园"}, ensure_ascii=False),
    }
    adapter = FakeAdapter(
        [
            ModelResult(kind="tool_calls", tool_calls=[call], text=""),
            _text_result("拙政园明天 07:30 开门。"),
        ]
    )
    gateway, _clock = _build({"primary": adapter}, burst=10, rate_limit_enabled=True)
    result = run_agent_loop(
        adapter=gateway,
        runtime=ToolRuntime(SCENIC_TOOLS),
        ctx=ExecutionContext(run_id="gw", user_id="u", channel="app", scene="presale"),
        system="你是景区售前客服。",
        user="拙政园明天几点开门？",
        guard=LoopGuard(max_turns=4),
        on_event=lambda *_args: None,
    )

    assert result.stop_reason == "protocol_stop"
    assert result.text == "拙政园明天 07:30 开门。"
    assert len(gateway.records) == 2
    assert all(record.provider == "primary" for record in gateway.records)
    tool_messages = [item for item in result.messages if item.get("role") == "tool"]
    assert tool_messages[0]["tool_call_id"] == "call_hours_1"


def test_retry_policy_caps_backoff():
    policy = RetryPolicy(max_attempts_per_route=3, base_delay_seconds=0.2, max_delay_seconds=0.3)
    assert policy.delay_after(1) == pytest.approx(0.2)
    assert policy.delay_after(2) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts_per_route=2, base_delay_seconds=1, max_delay_seconds=0.2)


def test_classify_status_timeout_and_explicit_upstream_error():
    class HasStatus(Exception):
        def __init__(self) -> None:
            super().__init__("upstream 503")
            self.status_code = 503

    classified = classify_exception(HasStatus(), frozenset({"503", "timeout"}))
    assert classified.code == "503"
    assert classified.retryable is True

    timeout = classify_exception(TimeoutError("slow"), frozenset({"timeout"}))
    assert timeout.code == "timeout"
    assert timeout.retryable is True

    unknown = classify_exception(ValueError("bug"), frozenset({"500"}))
    assert unknown.code == "upstream_error"
    assert unknown.retryable is False

    explicit = UpstreamError("限流", code="429", retryable=True)
    assert classify_exception(explicit, frozenset()) is explicit


def test_config_deep_merge_keeps_gateway_defaults():
    """环境文件只改限流或冷却时，重试次数和另一侧字段还在。"""
    test_cfg = load_config("test")
    prod_cfg = load_config("production")

    assert test_cfg.prompt.label == "test"
    assert prod_cfg.prompt.label == "production"
    assert test_cfg.gateway.default_alias == "default"
    assert test_cfg.gateway.rate_limit.requests_per_minute == 120
    assert test_cfg.gateway.circuit_breaker.failure_threshold == 3
    assert test_cfg.gateway.circuit_breaker.cooldown_seconds == 30
    assert test_cfg.gateway.retry.max_attempts_per_route == 2
    assert test_cfg.gateway.observe.langfuse_enabled is True
    assert prod_cfg.gateway.observe.langfuse_enabled is True
    assert prod_cfg.gateway.rate_limit.requests_per_minute == 60
    assert prod_cfg.gateway.circuit_breaker.failure_threshold == 5
    assert prod_cfg.gateway.circuit_breaker.cooldown_seconds == 45
    assert prod_cfg.gateway.retry.max_delay_seconds == 2
    assert [route.provider for route in test_cfg.gateway.aliases["default"].routes] == [
        "deepseek",
        "deepseek_backup",
    ]
    assert test_cfg.gateway.aliases["default"].strategy == "priority"
    fast = test_cfg.gateway.aliases["fast"]
    assert fast.strategy == "weighted_round_robin"
    assert [(route.provider, route.weight) for route in fast.routes] == [
        ("deepseek", 3),
        ("deepseek_backup", 1),
    ]


def test_config_fail_fast_on_bad_route_and_strategy():
    raw = load_config("test").gateway.model_dump()
    broken_provider = json.loads(json.dumps(raw))
    broken_provider["aliases"]["default"]["routes"][0]["provider"] = "missing"
    with pytest.raises(ValidationError):
        GatewayConfig.model_validate(broken_provider)

    broken_strategy = json.loads(json.dumps(raw))
    broken_strategy["aliases"]["default"]["strategy"] = "shuffle"
    with pytest.raises(ValidationError):
        GatewayConfig.model_validate(broken_strategy)

    broken_weight = json.loads(json.dumps(raw))
    broken_weight["aliases"]["fast"]["routes"][0]["weight"] = 0
    with pytest.raises(ValidationError):
        GatewayConfig.model_validate(broken_weight)

    broken_delay = json.loads(json.dumps(raw))
    broken_delay["retry"]["base_delay_seconds"] = 5
    broken_delay["retry"]["max_delay_seconds"] = 1
    with pytest.raises(ValidationError):
        GatewayConfig.model_validate(broken_delay)


def test_factory_uses_injected_adapters_and_rejects_unknown_names():
    cfg = load_config("test")
    primary = FakeAdapter(text="工厂主路由")
    backup = FakeAdapter(text="工厂备路由")
    gateway = build_gateway_from_config(
        cfg,
        adapters={"deepseek": primary, "deepseek_backup": backup},
    )

    result = gateway.complete(ModelRequest(system="s", user="你好"))
    assert result.text == "工厂主路由"
    assert backup.calls == 0

    with pytest.raises(ConfigError):
        build_gateway_from_config(cfg, adapters={"no-such": FakeAdapter()})


def test_factory_without_api_key_does_not_call_vendor(monkeypatch: pytest.MonkeyPatch):
    """清空密钥后构建网关，调用应立刻 missing_api_key，不能去 new 真实客户端。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = load_config("test")
    cleared = {
        name: provider.model_copy(update={"api_key": None})
        for name, provider in cfg.gateway.providers.items()
    }
    cfg = cfg.model_copy(update={"gateway": cfg.gateway.model_copy(update={"providers": cleared})})
    gateway = build_gateway_from_config(cfg)

    with pytest.raises(AllRoutesFailed) as caught:
        gateway.complete(ModelRequest(system="s", user="hi"))
    assert caught.value.code == "missing_api_key"
    record = gateway.records[-1]
    assert record.attempts == 2
    assert record.retries == 0
    assert record.fallback is True


def test_weighted_round_robin_smooth_distribution():
    """weight 2:1 时，6 次调用的首选顺序是平滑的，不是先把大权重连续打完。"""
    heavy = FakeAdapter(text="heavy")
    light = FakeAdapter(text="light")
    gateway, _clock = _build(
        {"heavy": heavy, "light": light},
        strategy="weighted_round_robin",
        weights={"heavy": 2, "light": 1},
        max_attempts=1,
        burst=6,
    )

    picked = []
    for _ in range(6):
        gateway.complete(_request())
        picked.append(gateway.records[-1].provider)

    assert picked == ["heavy", "light", "heavy", "heavy", "light", "heavy"]
    assert picked.count("heavy") == 4
    assert picked.count("light") == 2
    assert all(record.ttft_ms is None for record in gateway.records)


def test_priority_ignores_weight():
    """priority 永远先走配置里的第一条，哪怕第二条的 weight 更大。"""
    first = FakeAdapter(text="first")
    second = FakeAdapter(text="second")
    gateway, _clock = _build(
        {"first": first, "second": second},
        strategy="priority",
        weights={"first": 1, "second": 100},
        max_attempts=1,
        burst=3,
    )
    for _ in range(3):
        assert gateway.complete(_request()).text == "first"
    assert second.calls == 0


def test_weighted_round_robin_falls_back_when_chosen_route_fails():
    """这一次先选中备路由；它失败后仍会改打另一条，并记 fallback。"""
    primary = FakeAdapter(text="主路由兜底")
    backup = FakeAdapter([UpstreamError("备路由挂了", code="500", retryable=False)])
    gateway, _clock = _build(
        {"primary": primary, "backup": backup},
        strategy="weighted_round_robin",
        weights={"primary": 1, "backup": 2},
        max_attempts=1,
    )

    result = gateway.complete(_request())

    assert result.text == "主路由兜底"
    assert backup.calls == 1
    assert primary.calls == 1
    record = gateway.records[-1]
    assert record.fallback is True
    assert record.provider == "primary"


def test_unknown_strategy_fails_fast():
    with pytest.raises(GatewayError, match="weighted_round_robin"):
        _build({"primary": FakeAdapter()}, strategy="shuffle")


def test_stream_ttft_is_none_until_first_event_and_includes_retry_wait():
    """非流式没有 TTFT。流式在第一条事件之前的退避要算进 ttft。"""

    class Immediate(FakeAdapter):
        def stream(self, request: ModelRequest, *, cancel_event: threading.Event | None = None):
            del request, cancel_event
            self.calls += 1
            yield ModelStreamEvent(type="text.delta", data={"delta": "你"})

    immediate = Immediate()
    gateway, _clock = _build({"primary": immediate}, max_attempts=1)
    assert gateway.complete(_request()).text
    assert gateway.records[-1].ttft_ms is None

    events = list(gateway.stream(_request()))
    assert events[0].data["delta"] == "你"
    assert gateway.records[-1].stream is True
    assert gateway.records[-1].ttft_ms == 0.0

    class Flaky(FakeAdapter):
        def stream(self, request: ModelRequest, *, cancel_event: threading.Event | None = None):
            del request, cancel_event
            self.calls += 1
            if self.calls == 1:
                raise UpstreamError("超时", code="timeout", retryable=True)
            yield ModelStreamEvent(type="text.delta", data={"delta": "好"})

    flaky = Flaky()
    retry_gateway, clock = _build(
        {"primary": flaky},
        max_attempts=2,
        base_delay=0.2,
        max_delay=1.0,
    )
    assert list(retry_gateway.stream(_request()))[0].data["delta"] == "好"
    assert clock.slept == [0.2]
    assert retry_gateway.records[-1].ttft_ms == pytest.approx(200.0)
    assert flaky.calls == 2

    class Boom(FakeAdapter):
        def stream(self, request: ModelRequest, *, cancel_event: threading.Event | None = None):
            del request, cancel_event
            self.calls += 1
            raise UpstreamError("没连上", code="503", retryable=False)
            yield ModelStreamEvent(type="text.delta", data={"delta": "不会执行"})

    dead = Boom()
    dead_gateway, _clock = _build({"primary": dead}, max_attempts=1)
    with pytest.raises(AllRoutesFailed):
        list(dead_gateway.stream(_request()))
    assert dead_gateway.records[-1].stream is True
    assert dead_gateway.records[-1].ttft_ms is None


class _FakeLangfuse:
    """不连网。start_observation 把参数留下来，end 记一次。"""

    def __init__(self) -> None:
        self.started: list[dict] = []
        self.ended = 0
        self.fail = False

    def start_observation(self, **kwargs):
        if self.fail:
            raise RuntimeError("langfuse down")
        self.started.append(kwargs)
        return self

    def end(self):
        self.ended += 1
        return self


def test_langfuse_sink_is_noop_without_keys(monkeypatch: pytest.MonkeyPatch):
    """没配密钥时不能去 new Langfuse，即使开关是开的。"""

    def boom(*_args, **_kwargs):
        raise AssertionError("缺密钥时不应创建 Langfuse 客户端")

    monkeypatch.setattr("langfuse.Langfuse", boom)
    cfg = load_config("test")
    cfg = cfg.model_copy(
        update={
            "langfuse": cfg.langfuse.model_copy(update={"public_key": None, "secret_key": None})
        }
    )
    gateway = build_gateway_from_config(
        cfg,
        adapters={
            "deepseek": FakeAdapter(text="不上报"),
            "deepseek_backup": FakeAdapter(text="备"),
        },
    )
    assert gateway.complete(ModelRequest(system="s", user="hi")).text == "不上报"


def test_langfuse_sink_records_call_and_can_be_disabled():
    cfg = load_config("test")
    client = _FakeLangfuse()
    seen: list[object] = []
    gateway = build_gateway_from_config(
        cfg,
        adapters={
            "deepseek": FakeAdapter(text="已记录"),
            "deepseek_backup": FakeAdapter(text="备"),
        },
        langfuse_client=client,
        on_record=seen.append,
    )
    assert gateway.complete(ModelRequest(system="s", user="hi")).text == "已记录"
    assert len(client.started) == 1
    assert client.ended == 1
    meta = client.started[0]["metadata"]
    assert client.started[0]["name"] == "gateway.complete"
    assert client.started[0]["as_type"] == "generation"
    assert client.started[0]["model"] == "deepseek-chat"
    assert meta["alias"] == "default"
    assert meta["provider"] == "deepseek"
    assert meta["fallback"] is False
    assert meta["retries"] == 0
    assert meta["ttft_ms"] is None
    assert meta["error_code"] is None
    assert "上次输出" not in str(client.started[0])
    assert client.started[0]["usage_details"]["input"] == 1
    assert seen and seen[0].provider == "deepseek"  # type: ignore[attr-defined]

    client.fail = True
    assert gateway.complete(ModelRequest(system="s", user="再来")).text == "已记录"

    disabled = cfg.model_copy(
        update={
            "gateway": cfg.gateway.model_copy(
                update={
                    "observe": cfg.gateway.observe.model_copy(update={"langfuse_enabled": False})
                }
            )
        }
    )
    quiet = _FakeLangfuse()
    off = build_gateway_from_config(
        disabled,
        adapters={
            "deepseek": FakeAdapter(text="关掉了"),
            "deepseek_backup": FakeAdapter(text="备"),
        },
        langfuse_client=quiet,
    )
    assert off.complete(ModelRequest(system="s", user="hi")).text == "关掉了"
    assert quiet.started == []
    assert quiet.ended == 0


def test_langfuse_stream_generation_includes_ttft():
    cfg = load_config("test")

    class Once(FakeAdapter):
        def stream(self, request: ModelRequest, *, cancel_event: threading.Event | None = None):
            del request, cancel_event
            yield ModelStreamEvent(type="text.delta", data={"delta": "流"})

    client = _FakeLangfuse()
    gateway = build_gateway_from_config(
        cfg,
        adapters={"deepseek": Once(), "deepseek_backup": FakeAdapter()},
        langfuse_client=client,
        clock=FakeClock(),
    )
    assert list(gateway.stream(ModelRequest(system="s", user="hi")))[0].data["delta"] == "流"
    assert client.started[0]["name"] == "gateway.stream"
    assert client.started[0]["metadata"]["ttft_ms"] == 0.0
    assert client.ended == 1

