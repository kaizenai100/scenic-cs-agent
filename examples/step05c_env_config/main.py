"""步骤 5.6 · 按环境加载 yaml 配置，prompt label 区分 test / production。

跑法：
  APP_ENV=test       uv run python examples/step05c_env_config/main.py
  APP_ENV=production uv run python examples/step05c_env_config/main.py

读代码盯：
  1) config/project.yaml + project_<env>.yaml — 公共默认 + 环境差异
  2) src/scenic_cs_agent/config/loader.py     — 选环境、深合并、${VAR} 占位、Pydantic 校验
  3) prompts/registry.py                       — label / 缓存时间来自配置，不再写死

本示例演示：
  - 两个环境实际生效的配置差异（同一份代码，行为只由配置决定）
  - 当前环境拉到的 prompt 来源、label、版本号
  - 非法环境名 → 启动直接报错（不悄悄回落默认）
"""

from __future__ import annotations

from scenic_cs_agent.config import ConfigError, get_config, load_config
from scenic_cs_agent.prompts import render_scenic_presale


def show_env_diff() -> None:
    """不依赖 APP_ENV，显式加载两个环境，对比差异项。密钥只显示是否配置，不打印值。"""
    print("=== 两个环境的配置差异")
    for env in ("test", "production"):
        c = load_config(env)
        print(
            f"[{c.env:<10}] prompt.label={c.prompt.label:<10} "
            f"cache_ttl={c.prompt.cache_ttl_seconds:<3} "
            f"langfuse.environment={c.langfuse.environment:<10} "
            f"model.api_key={'已配置' if c.model.api_key else '未配置'}"
        )


def show_current_prompt() -> None:
    """按当前 APP_ENV 取 prompt：业务代码不传 label，完全跟随配置。"""
    cfg = get_config()
    p = render_scenic_presale()
    print(f"\n=== 当前环境 APP_ENV={cfg.env}")
    print(f"source={p.source} label={p.label} version={p.version} fingerprint={p.fingerprint[:12]}")


def show_bad_env() -> None:
    """环境名拼错必须立刻报错：生产机器上跑着测试配置是最难发现的事故。"""
    print("\n=== 非法环境名")
    try:
        load_config("prodution")  # 故意拼错
    except ConfigError as exc:
        print(f"ConfigError: {exc}")


if __name__ == "__main__":
    show_env_diff()
    show_current_prompt()
    show_bad_env()
