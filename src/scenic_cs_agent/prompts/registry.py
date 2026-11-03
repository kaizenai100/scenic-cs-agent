"""Prompt 来源：Langfuse 为主，本地文件兜底；拉哪个 label 由环境配置决定。

发布 / 回滚靠 Langfuse 的 label：
  - 每次保存 = 一个不可变版本（v1, v2, ...）
  - label 指向哪个版本，对应环境就用哪个：
      test 环境拉 label "test"，生产拉 label "production"（见 config/project_*.yaml）
  - 发布流程：新版本先只打 test → 测试环境验证 → 把 production 挪到同一版本
  - 回滚 = 把 production 挪回旧版本，不改代码、不发版

为什么 label 放配置而不是写死在代码里：
  同一份代码部署到不同环境，行为差异只来自配置；代码里写死 "production"
  会让测试环境直接吃线上 prompt，新 prompt 就没地方先验证。

实时生效：SDK 本地缓存 cache_ttl_seconds 秒（按环境配置），过期后台刷新；
Langfuse 挂了 → 用本地同名 .txt 兜底，业务不中断。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from langfuse import Langfuse

from scenic_cs_agent.config import get_config
from scenic_cs_agent.prompts.render import RenderedPrompt, fingerprint, render_template

PROMPTS_DIR = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


def local_template(name: str) -> str:
    """本地兜底模板：与 Langfuse production 版内容保持一致，只此一份（不分环境）。"""
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _client() -> Langfuse:
    """Langfuse 客户端单例：连接信息和环境标签全部来自配置，不再直接读 os.environ。"""
    cfg = get_config().langfuse
    return Langfuse(
        host=cfg.host,
        public_key=cfg.public_key,
        secret_key=cfg.secret_key,
        environment=cfg.environment,  # trace 按环境区分，测试流量不污染生产看板
    )


def get_prompt(
    name: str,
    variables: dict[str, str],
    *,
    label: str | None = None,
) -> RenderedPrompt:
    """取 prompt 并渲染。

    label 默认 None = 用当前环境配置的 label；显式传值只用于演示 / 排查
    （比如故意传一个不存在的 label 看兜底），业务代码不要传。
    """
    pcfg = get_config().prompt
    label = label or pcfg.label
    fallback = local_template(name)
    try:
        p = _client().get_prompt(
            name,
            label=label,
            type="text",
            fallback=fallback,
            cache_ttl_seconds=pcfg.cache_ttl_seconds,
            fetch_timeout_seconds=pcfg.fetch_timeout_seconds,
            max_retries=0,  # 拉不到就兜底，不在对话链路上重试拖延时延
        )
        template, is_fallback, version = p.prompt, p.is_fallback, str(p.version)
    except Exception:
        # SDK 自身异常也兜底（比如 key 没配）
        template, is_fallback, version = fallback, True, "local"

    if is_fallback:
        # 兜底不是正常状态：测试环境拉不到 test label 往往是忘了打 label，要能在日志里看到
        logger.warning("prompt %s label=%s 拉取失败，使用本地兜底", name, label)

    # 只用自己的渲染器，不用 p.compile()，保证规则唯一且严格
    text = render_template(template, variables)
    return RenderedPrompt(
        text=text,
        name=name,
        label=label,  # 记录实际使用的 label，排查“这次用的是哪个环境的 prompt”
        version="local" if is_fallback else version,
        source="local-fallback" if is_fallback else "langfuse",
        fingerprint=fingerprint(text),
    )
