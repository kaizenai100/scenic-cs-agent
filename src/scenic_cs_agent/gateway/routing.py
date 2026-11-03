"""别名 → 有序路由。

业务只说「我要 default / fast」。具体打哪家、哪个模型写在配置的路由表里。
strategy 目前只支持 priority：按列表顺序试，前面的失败或熔断打开才试下一条。
这就是 fallback 列表，不在这里做权重轮询——权重要先有真实多上游流量再加，避免骨架里一套用不上的调度。
"""

from __future__ import annotations

from dataclasses import dataclass

from scenic_cs_agent.gateway.errors import GatewayError
from scenic_cs_agent.runtime.model_types import ModelAdapter


@dataclass(frozen=True)
class RouteSpec:
    """配置里的一条路由。model 必须和该 provider 绑定的模型一致，防止写错还静默打到另一个模型。"""

    provider: str
    model: str


@dataclass(frozen=True)
class AliasSpec:
    strategy: str
    routes: tuple[RouteSpec, ...]


@dataclass
class ProviderBinding:
    """一个已注册的上游。adapter 是厂商差异所在，Gateway 不关心它内部是 Chat 还是 Responses。"""

    name: str
    adapter: ModelAdapter
    model: str
    enabled: bool = True


@dataclass(frozen=True)
class ResolvedRoute:
    """解析之后可以直接调用的路由。"""

    provider: str
    model: str
    adapter: ModelAdapter


class Router:
    """启动时做完校验。请求路径上只查表，不再猜配置是否合法。"""

    def __init__(
        self,
        *,
        bindings: dict[str, ProviderBinding],
        aliases: dict[str, AliasSpec],
        default_alias: str,
    ) -> None:
        if not bindings:
            raise GatewayError("至少要注册一个 provider", code="no_provider", retryable=False)
        if default_alias not in aliases:
            raise GatewayError(
                f"default_alias={default_alias!r} 不在 aliases 里",
                code="bad_config",
                retryable=False,
            )
        for alias, spec in aliases.items():
            if spec.strategy != "priority":
                raise GatewayError(
                    f"别名 {alias} 的 strategy={spec.strategy!r} 不支持，目前只支持 priority",
                    code="bad_config",
                    retryable=False,
                )
            if not spec.routes:
                raise GatewayError(f"别名 {alias} 的路由列表为空", code="bad_config", retryable=False)
            usable = 0
            for route in spec.routes:
                binding = bindings.get(route.provider)
                if binding is None or not binding.enabled:
                    # 禁用的 provider 可以留在列表里，运行时跳过；但别名不能一条可用路由都没有。
                    continue
                usable += 1
                if route.model != binding.model:
                    raise GatewayError(
                        f"别名 {alias} 的路由 {route.provider} model={route.model!r} "
                        f"与 provider.model={binding.model!r} 不一致",
                        code="bad_config",
                        retryable=False,
                    )
            if usable == 0:
                raise GatewayError(f"别名 {alias} 没有可用路由", code="no_route", retryable=False)
        self._bindings = bindings
        self._aliases = aliases
        self._default_alias = default_alias

    @property
    def default_alias(self) -> str:
        return self._default_alias

    def routes_for(self, alias: str | None) -> list[ResolvedRoute]:
        """按 priority 返回本次要尝试的路由。未知别名直接失败，不偷偷改走 default。"""
        name = alias or self._default_alias
        spec = self._aliases.get(name)
        if spec is None:
            raise GatewayError(f"未知模型别名 {name!r}", code="unknown_alias", retryable=False)
        resolved: list[ResolvedRoute] = []
        for route in spec.routes:
            binding = self._bindings.get(route.provider)
            if binding is None or not binding.enabled:
                continue
            resolved.append(
                ResolvedRoute(provider=binding.name, model=binding.model, adapter=binding.adapter)
            )
        if not resolved:
            raise GatewayError(f"别名 {name} 没有可用路由", code="no_route", retryable=False)
        return resolved
