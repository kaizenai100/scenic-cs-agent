"""从现有 yaml 配置组装 Gateway。

业务代码拿模型的方式：build_gateway_from_config()，然后只调 gateway.complete / stream。
不要在业务里读 DEEPSEEK_API_KEY 再自己 new 客户端，也不要在业务里写重试。

测试和示例可以传入 adapters，用 Fake 替换掉真实适配器。
这样没配密钥也能把「配置 → 路由 → 调用」这条路径跑通，而且不会发出网络请求。
注入的名字必须是配置里的 provider，写错在启动时失败，而不是调用时才发现打到了真上游。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from scenic_cs_agent.config.loader import AppConfig, ConfigError, get_config
from scenic_cs_agent.gateway.clock import Clock
from scenic_cs_agent.gateway.gateway import LLMGateway
from scenic_cs_agent.gateway.observe import CallRecord
from scenic_cs_agent.gateway.retry import RetryPolicy
from scenic_cs_agent.gateway.routing import AliasSpec, ProviderBinding, RouteSpec
from scenic_cs_agent.gateway.unconfigured import UnconfiguredAdapter
from scenic_cs_agent.runtime.model_types import ModelAdapter


def build_gateway_from_config(
    config: AppConfig | None = None,
    *,
    adapters: Mapping[str, ModelAdapter] | None = None,
    clock: Clock | None = None,
    on_record: Callable[[CallRecord], None] | None = None,
) -> LLMGateway:
    """按配置注册 provider 和别名。config 为空时用进程内的 get_config()。"""
    app = config if config is not None else get_config()
    gateway_cfg = app.gateway
    injected = dict(adapters or {})
    unknown = sorted(set(injected) - set(gateway_cfg.providers))
    if unknown:
        raise ConfigError(f"注入的适配器没有对应 provider：{', '.join(unknown)}")

    bindings: dict[str, ProviderBinding] = {}
    for name, provider in gateway_cfg.providers.items():
        if not provider.enabled:
            # 禁用的上游不注册。路由表里若还写着它，Router 会跳过。
            continue
        if name in injected:
            adapter: ModelAdapter = injected[name]
        elif not provider.api_key:
            # 没密钥就不建真实客户端。调用到这条路由时给出 missing_api_key，而不是连到上游。
            adapter = UnconfiguredAdapter(name)
        else:
            adapter = _create_vendor_adapter(provider)
        bindings[name] = ProviderBinding(name=name, adapter=adapter, model=provider.model)

    aliases: dict[str, AliasSpec] = {}
    for alias, spec in gateway_cfg.aliases.items():
        routes: list[RouteSpec] = []
        for route in spec.routes:
            provider = gateway_cfg.providers[route.provider]
            model = route.model or provider.model
            routes.append(RouteSpec(provider=route.provider, model=model))
        aliases[alias] = AliasSpec(strategy=spec.strategy, routes=tuple(routes))

    retry_cfg = gateway_cfg.retry
    circuit_cfg = gateway_cfg.circuit_breaker
    limit_cfg = gateway_cfg.rate_limit
    return LLMGateway(
        bindings=bindings,
        aliases=aliases,
        default_alias=gateway_cfg.default_alias,
        retry=RetryPolicy(
            max_attempts_per_route=retry_cfg.max_attempts_per_route,
            base_delay_seconds=retry_cfg.base_delay_seconds,
            max_delay_seconds=retry_cfg.max_delay_seconds,
            retryable_codes=frozenset(retry_cfg.retryable_codes),
        ),
        clock=clock,
        failure_threshold=circuit_cfg.failure_threshold,
        cooldown_seconds=circuit_cfg.cooldown_seconds,
        half_open_max_probes=circuit_cfg.half_open_max_probes,
        rate_limit_enabled=limit_cfg.enabled,
        requests_per_minute=limit_cfg.requests_per_minute,
        burst=limit_cfg.burst,
        structured_output_retries=gateway_cfg.structured_output_retries,
        on_record=on_record,
    )


def _create_vendor_adapter(provider: object) -> ModelAdapter:
    """按 kind 选择适配器类。kind 决定协议，model / base_url / key 决定打到哪。"""
    # 局部导入：没走到真上游时，不强制每个调用方都在导入 Gateway 时创建客户端。
    from scenic_cs_agent.config.loader import ProviderConfig
    from scenic_cs_agent.runtime.deepseek_chat_adapter import DeepSeekChatAdapter
    from scenic_cs_agent.runtime.deepseek_responses_adapter import DeepSeekResponsesAdapter
    from scenic_cs_agent.runtime.openai_responses_adapter import OpenAIResponsesAdapter

    if not isinstance(provider, ProviderConfig):
        raise ConfigError("provider 配置类型不正确")
    if provider.kind == "deepseek_chat":
        return DeepSeekChatAdapter(
            api_key=provider.api_key,
            model=provider.model,
            base_url=provider.base_url or "https://api.deepseek.com",
            timeout=provider.timeout_seconds,
        )
    if provider.kind == "deepseek_responses":
        return DeepSeekResponsesAdapter(
            api_key=provider.api_key,
            model=provider.model,
            base_url=provider.base_url or "https://api.deepseek.com",
            timeout=provider.timeout_seconds,
        )
    if provider.kind == "openai_responses":
        return OpenAIResponsesAdapter(
            api_key=provider.api_key,
            model=provider.model,
            base_url=provider.base_url,
            timeout=provider.timeout_seconds,
        )
    raise ConfigError(f"不支持的适配器 kind={provider.kind!r}")
