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
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

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


class AppConfig(BaseModel):
    """整个应用的配置根对象。frozen：运行期不允许被某个模块偷偷改掉。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    env: str  # 实际生效的环境名，打日志/排查时用
    project_name: str
    langfuse: LangfuseConfig
    prompt: PromptConfig
    model: ModelConfig


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
