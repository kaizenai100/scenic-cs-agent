"""别名 → 这次要尝试的路由顺序。

业务只说「我要 default / fast」。具体打哪家、哪个模型写在配置的路由表里。

两种策略：

  priority
    默认。永远按配置顺序试：第一条是主路由，失败或熔断打开才用下一条。
    路由上的 weight 可以写，但 priority 不看它。这样同一张表改策略时不用先删字段。

  weighted_round_robin
    按 weight 决定「这一次先打谁」。weight 越大，被选中的次数越多。
    选中的那条放在最前，后面仍按配置里的相对顺序做 fallback，所以权重调度和失败转移是两件事。
    用的是平滑加权（每次给每条加上自己的权重，选当前最大的，再减去总权重）。
    不把权重展开成「连续打同一家 weight 次」：那样一家抖动时，接下来一整段请求都会打进故障上游。

未知策略在组装 Router 时直接失败，不等到请求进来再猜。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from scenic_cs_agent.gateway.errors import GatewayError
from scenic_cs_agent.runtime.model_types import ModelAdapter

_STRATEGIES = frozenset({"priority", "weighted_round_robin"})


@dataclass(frozen=True)
class RouteSpec:
    """配置里的一条路由。model 必须和该 provider 绑定的模型一致，防止写错还静默打到另一个模型。"""

    provider: str
    model: str
    # 省略时工厂会填 1。priority 不消费这个数；加权轮询要求它 >= 1。
    weight: int = 1


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
    """解析之后可以直接调用的路由。weight 只给加权调度用。"""

    provider: str
    model: str
    adapter: ModelAdapter
    weight: int = 1


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
            if spec.strategy not in _STRATEGIES:
                raise GatewayError(
                    f"别名 {alias} 的 strategy={spec.strategy!r} 不支持，"
                    "可选 priority / weighted_round_robin",
                    code="bad_config",
                    retryable=False,
                )
            if not spec.routes:
                raise GatewayError(f"别名 {alias} 的路由列表为空", code="bad_config", retryable=False)
            usable = 0
            for route in spec.routes:
                if route.weight < 1:
                    raise GatewayError(
                        f"别名 {alias} 的路由 {route.provider} weight 必须 >= 1",
                        code="bad_config",
                        retryable=False,
                    )
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
        # 每个别名自己的当前权重。别名之间不共享，否则 fast 的调度会被 default 的调用打乱。
        self._wrr_current: dict[str, dict[str, int]] = {}
        self._wrr_lock = threading.Lock()

    @property
    def default_alias(self) -> str:
        return self._default_alias

    def routes_for(self, alias: str | None, *, schedule: bool = True) -> list[ResolvedRoute]:
        """返回这次要尝试的路由，第一条是首选。

        schedule=False 只按配置顺序看有哪些路由，不推进加权计数。
        网关初始化时拿它读适配器能力，那一次不能算作用户请求。
        未知别名直接失败，不偷偷改走 default。
        """
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
                ResolvedRoute(
                    provider=binding.name,
                    model=binding.model,
                    adapter=binding.adapter,
                    weight=route.weight,
                )
            )
        if not resolved:
            raise GatewayError(f"别名 {name} 没有可用路由", code="no_route", retryable=False)
        if not schedule or spec.strategy == "priority":
            return resolved
        start = self._pick_weighted(name, resolved)
        # 从被选中的那条开始，后面仍按原相对顺序兜底，绕回配置表的前面。
        return resolved[start:] + resolved[:start]

    def _pick_weighted(self, alias: str, routes: list[ResolvedRoute]) -> int:
        """平滑加权：返回本次首选在 routes 里的下标。"""
        with self._wrr_lock:
            state = self._wrr_current.setdefault(alias, {})
            alive = {self._route_key(index, route) for index, route in enumerate(routes)}
            # 路由表变了（比如禁用了某条）就丢掉旧计数，避免一个已经不存在的 key 一直占着权重。
            for stale in list(state):
                if stale not in alive:
                    del state[stale]
            total = sum(route.weight for route in routes)
            best_index = 0
            best_current: int | None = None
            for index, route in enumerate(routes):
                key = self._route_key(index, route)
                state[key] = state.get(key, 0) + route.weight
                current = state[key]
                # 严格大于：权重相同时留在配置里更靠前的那条，结果可重复。
                if best_current is None or current > best_current:
                    best_current = current
                    best_index = index
            chosen = routes[best_index]
            state[self._route_key(best_index, chosen)] -= total
            return best_index

    @staticmethod
    def _route_key(index: int, route: ResolvedRoute) -> str:
        return f"{index}:{route.provider}:{route.model}"
