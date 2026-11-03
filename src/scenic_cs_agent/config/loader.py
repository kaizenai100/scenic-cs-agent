"""按环境加载 yaml 配置：project.yaml（公共默认）+ project_<env>.yaml（环境覆盖）。

为什么要有这个模块：
  - 之前配置散在各处 os.environ / 函数默认参数里（比如 prompt label 写死 "production"），
    测试和生产没法区分，改一个值要翻好几个文件。
  - 现在统一成：一个入口 get_config()，所有模块只从这里取配置。

设计要点：
  1. 环境由 APP_ENV 决定，默认 test。不认识的值直接报错，而不是悄悄回落默认——
     生产机器上 APP_ENV 拼错却跑着测试配置，是最难发现的事故。
  2. 公共文件 + 环境文件“深合并”：环境文件只写差异项，减少多份配置复制后漂移。
  3. 密钥不进 yaml（公开仓），用 ${VAR} / ${VAR:-默认} 占位，加载时从环境变量替换；
     必填的没配就启动报错（fail fast）。
  4. 合并后用 Pydantic 校验成强类型对象：字段拼错、类型不对，启动时就暴露。
  5. lru_cache 保证进程内只加载一次（单例）；测试里切环境时调 get_config.cache_clear()。
  6. gateway 段同样走这套合并和校验。别名、重试、熔断、限流写在 yaml 里，
     业务代码只调 Gateway，不在调用点再读一遍上游密钥。
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# 仓库根目录下的 config/：src/scenic_cs_agent/config/loader.py 往上 4 层就是仓库根
CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"

# 合法环境；别名统一成标准名，免得 prod / production 两套写法各配一份
VALID_ENVS = {"test", "production"}
ENV_ALIASES = {"prod": "production", "product": "production", "dev": "test"}

# 匹配 ${VAR} 或 ${VAR:-默认值}；默认值部分允许为空字符串
_PLACEHOLDER = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


class ConfigError(RuntimeError):
    """配置错误：环境名不合法、文件缺失、必填环境变量没配。启动阶段抛出，不吞。"""


# ---------------- 强类型配置结构 ----------------
# extra="forbid"：yaml 里多写/拼错一个 key 直接报错，避免“以为配上了其实没生效”


class LangfuseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str
    public_key: str | None = None  # 允许缺：缺了 prompt 走本地兜底，不阻断启动
    secret_key: str | None = None
    environment: str  # trace 上的环境标签


class PromptConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str  # 拉 Langfuse 哪个 label：test / production
    cache_ttl_seconds: int = 30
    fetch_timeout_seconds: int = 3


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    base_url: str
    api_key: str | None = None  # 只跑 prompt 示例时可以不配模型 key
    timeout_seconds: float = 60


def _default_retryable_codes() -> list[str]:
    # 超时、连接失败、429、常见 5xx。408/409 不默认重试，需要时在 yaml 里加。
    return ["timeout", "connection", "429", "500", "502", "503", "504"]


class RetryConfig(BaseModel):
    """同一条路由上的重试预算。max_attempts_per_route 含第一次调用。"""

    model_config = ConfigDict(extra="forbid")
    max_attempts_per_route: int = Field(ge=1, le=8)
    base_delay_seconds: float = Field(ge=0, le=60)
    max_delay_seconds: float = Field(ge=0, le=120)
    retryable_codes: list[str] = Field(default_factory=_default_retryable_codes)

    @field_validator("retryable_codes")
    @classmethod
    def _codes_not_empty(cls, value: list[str]) -> list[str]:
        # yaml 里如果写成数字，宽松模式会转成字符串；空串说明占位写坏了。
        cleaned = [str(item).strip() for item in value]
        if not cleaned or any(not item for item in cleaned):
            raise ValueError("gateway.retry.retryable_codes 不能为空")
        return cleaned

    @model_validator(mode="after")
    def _delay_order(self) -> Self:
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("gateway.retry.max_delay_seconds 不能小于 base_delay_seconds")
        return self


class CircuitBreakerConfig(BaseModel):
    """按 provider 熔断。冷却结束先半开，只放少量探测，成功才关上。"""

    model_config = ConfigDict(extra="forbid")
    failure_threshold: int = Field(ge=1, le=100)
    cooldown_seconds: float = Field(gt=0, le=3600)
    half_open_max_probes: int = Field(default=1, ge=1, le=5)


class RateLimitConfig(BaseModel):
    """进程内令牌桶。enabled=false 时保留参数，方便环境文件只改开关。"""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    requests_per_minute: int = Field(ge=1, le=100_000)
    burst: int = Field(ge=1, le=10_000)


class ProviderConfig(BaseModel):
    """一个上游。kind 选适配器类，model / base_url / api_key 选具体端点。

    api_key 允许为空：公开仓不写真实密钥，没配 ${VAR} 时启动仍可加载配置。
    真的打到这条路由才会以 missing_api_key 失败，而不是在导入阶段连上游。
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["deepseek_chat", "deepseek_responses", "openai_responses"]
    model: str = Field(min_length=1)
    base_url: str | None = None
    api_key: str | None = None
    timeout_seconds: float = Field(default=60, gt=0, le=600)
    enabled: bool = True


class RouteConfig(BaseModel):
    """别名下的一条路由。model 省略时用 provider.model；写了就必须一致，避免静默打错模型。

    weight 只被 weighted_round_robin 使用。priority 忽略它，但允许写在配置里，换策略时不用删字段。
    """

    model_config = ConfigDict(extra="forbid")
    provider: str = Field(min_length=1)
    model: str | None = None
    weight: int = Field(default=1, ge=1, le=10_000)

    @field_validator("model")
    @classmethod
    def _model_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("route.model 不能是空字符串")
        return value


class AliasConfig(BaseModel):
    """priority：按顺序试，前面的失败或熔断打开才用下一条。这是默认策略。

    weighted_round_robin：按 weight 决定这一次先打谁，失败后仍按其余路由 fallback。
    """

    model_config = ConfigDict(extra="forbid")
    strategy: Literal["priority", "weighted_round_robin"] = "priority"
    routes: list[RouteConfig] = Field(min_length=1)


class ObserveConfig(BaseModel):
    """网关观测开关。连接信息不用在这里再写一份，用上面的 langfuse 段。

    langfuse_enabled=true 且 public_key、secret_key 都有时，才把调用记成 generation。
    缺密钥或设成 false：不建客户端，调用照常进行。这里不拉 prompt。
    """

    model_config = ConfigDict(extra="forbid")
    langfuse_enabled: bool = True


class GatewayConfig(BaseModel):
    """进程内网关。业务只认别名，不在代码里散落上游地址和重试。"""

    model_config = ConfigDict(extra="forbid")
    default_alias: str = Field(min_length=1)
    structured_output_retries: int = Field(default=1, ge=0, le=5)
    retry: RetryConfig
    circuit_breaker: CircuitBreakerConfig
    rate_limit: RateLimitConfig
    observe: ObserveConfig = Field(default_factory=ObserveConfig)
    providers: dict[str, ProviderConfig]
    aliases: dict[str, AliasConfig]

    @model_validator(mode="after")
    def _routes_match_providers(self) -> Self:
        # 跨字段的错误放在启动校验里。请求路径上不再发现「别名指向一个不存在的上游」。
        if not self.providers:
            raise ValueError("gateway.providers 不能为空")
        if self.default_alias not in self.aliases:
            raise ValueError(f"gateway.default_alias={self.default_alias!r} 不在 aliases 里")
        for alias, spec in self.aliases.items():
            usable = 0
            for route in spec.routes:
                provider = self.providers.get(route.provider)
                if provider is None:
                    raise ValueError(f"别名 {alias} 引用了不存在的 provider {route.provider!r}")
                if route.model is not None and route.model != provider.model:
                    raise ValueError(
                        f"别名 {alias} 的路由 {route.provider} model={route.model!r} "
                        f"与 provider.model={provider.model!r} 不一致"
                    )
                if provider.enabled:
                    usable += 1
            if usable == 0:
                raise ValueError(f"别名 {alias} 没有可用路由（provider 都禁用了或不存在）")
        return self


class AppConfig(BaseModel):
    """整个应用的配置根对象。frozen：运行期不允许被某个模块偷偷改掉。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    env: str  # 实际生效的环境名，打日志/排查时用
    project_name: str
    langfuse: LangfuseConfig
    prompt: PromptConfig
    model: ModelConfig
    gateway: GatewayConfig


# ---------------- 加载流程 ----------------


def resolve_env(raw: str | None = None) -> str:
    """解析当前环境名：显式参数 > APP_ENV > 默认 test；别名归一；非法值报错。"""
    value = (raw or os.getenv("APP_ENV") or "test").strip().lower()
    value = ENV_ALIASES.get(value, value)
    if value not in VALID_ENVS:
        # 故意不回落默认：拼错环境名必须让人立刻知道
        raise ConfigError(f"APP_ENV={value!r} 不合法，可选 {sorted(VALID_ENVS)}")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    """读单个 yaml；文件为空时返回 {}，便于环境文件“什么都不覆盖”。"""
    if not path.exists():
        raise ConfigError(f"配置文件不存在：{path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}  # safe_load：不执行任意对象构造


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并：两边都是 dict 就往下合并，否则 override 直接覆盖。

    为什么不用 dict.update：update 是浅合并，环境文件写了 prompt.label，
    会把整个 prompt 段替换掉，cache_ttl_seconds 等默认值就丢了。
    """
    merged = dict(base)  # 复制一份，不修改入参
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value  # 列表、标量整体覆盖，语义最直观
    return merged


def _expand(value: Any, missing: list[str]) -> Any:
    """递归替换 ${VAR} 占位。缺失的必填变量收集到 missing，最后一次性报出来。"""
    if isinstance(value, dict):
        return {k: _expand(v, missing) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, missing) for v in value]
    if not isinstance(value, str):
        return value

    def repl(m: re.Match[str]) -> str:
        name, default = m.group(1), m.group(2)
        env_value = os.getenv(name)
        if env_value:
            return env_value
        if default is not None:
            return default
        missing.append(name)
        return ""

    expanded = _PLACEHOLDER.sub(repl, value)
    # 整个值就是一个没配的占位 → 返回 None，交给 Pydantic 判断该字段是否允许为空
    return None if expanded == "" and _PLACEHOLDER.fullmatch(value) else expanded


def load_config(env: str | None = None) -> AppConfig:
    """不带缓存的加载：测试或脚本想显式指定环境时用。"""
    load_dotenv()  # .env 里的密钥先进环境变量；已存在的系统变量不会被覆盖
    env = resolve_env(env)
    data = deep_merge(
        _read_yaml(CONFIG_DIR / "project.yaml"),
        _read_yaml(CONFIG_DIR / f"project_{env}.yaml"),
    )
    missing: list[str] = []
    data = _expand(data, missing)
    # 占位没配不立刻报错：有的字段允许为空（如 langfuse key）。
    # 真正必填的交给下面 Pydantic 校验；这里只记录，报错信息里带上方便排查。
    try:
        return AppConfig(env=env, **data)
    except Exception as exc:
        hint = f"（未配置的环境变量：{', '.join(missing)}）" if missing else ""
        raise ConfigError(f"[{env}] 配置校验失败{hint}：{exc}") from exc


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """进程内单例：业务代码统一调这个。切环境后需 get_config.cache_clear()。"""
    return load_config()
