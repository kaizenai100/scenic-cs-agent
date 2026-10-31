"""Prompt 来源：Langfuse 为主，本地文件兜底。

发布 / 回滚靠 Langfuse 的 label：
  - 每次保存 = 一个不可变版本（v1, v2, ...）
  - label "production" 指向哪个版本，线上就用哪个
  - 回滚 = 把 production 挪回旧版本，不改内容

实时生效：SDK 本地缓存 cache_ttl_seconds 秒，过期后台刷新；
Langfuse 挂了 → 用本地同名 .txt 兜底，业务不中断。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from langfuse import Langfuse

from scenic_cs_agent.prompts.render import RenderedPrompt, fingerprint, render_template

PROMPTS_DIR = Path(__file__).resolve().parent
DEFAULT_LABEL = "production"


def local_template(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _client() -> Langfuse:
    return Langfuse(
        host=os.environ.get("LANGFUSE_HOST", "http://localhost:3300"),
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
    )


def get_prompt(
    name: str,
    variables: dict[str, str],
    *,
    label: str = DEFAULT_LABEL,
    cache_ttl_seconds: int = 30,
) -> RenderedPrompt:
    fallback = local_template(name)
    try:
        p = _client().get_prompt(
            name,
            label=label,
            type="text",
            fallback=fallback,
            cache_ttl_seconds=cache_ttl_seconds,
            fetch_timeout_seconds=3,
            max_retries=0,
        )
        template, is_fallback, version = p.prompt, p.is_fallback, str(p.version)
    except Exception:
        # SDK 自身异常也兜底（比如 key 没配）
        template, is_fallback, version = fallback, True, "local"

    # 只用自己的渲染器，不用 p.compile()，保证规则唯一且严格
    text = render_template(template, variables)
    return RenderedPrompt(
        text=text,
        name=name,
        version="local" if is_fallback else version,
        source="local-fallback" if is_fallback else "langfuse",
        fingerprint=fingerprint(text),
    )
