"""Prompt 渲染：全仓只有一套规则 —— {{var}} 简单占位，没有 if/for。

为什么不用 Jinja2：
- prompt 主源头在 Langfuse，Langfuse 只支持 {{var}}
- 本地兜底文件和 Langfuse 内容一致，规则也必须一致，不能两套
- 条件逻辑（比如有没有工具）放 Python 拼好，再当变量传入，可单测

严格模式（等价 StrictUndefined）：
- 模板里有、调用没传 → 报错
- 调用传了、模板里没有 → 报错（防止传错名字静默失效）
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256

_VAR = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


class PromptRenderError(ValueError):
    pass


@dataclass(frozen=True)
class RenderedPrompt:
    """一次渲染结果：正文 + 来源 + 版本 + 指纹。"""

    text: str
    name: str
    version: str  # Langfuse 版本号，兜底时为 "local"
    source: str  # "langfuse" | "local-fallback"
    fingerprint: str


def render_template(template: str, variables: dict[str, str]) -> str:
    needed = set(_VAR.findall(template))
    missing = needed - variables.keys()
    extra = variables.keys() - needed
    if missing:
        raise PromptRenderError(f"模板变量未传入: {sorted(missing)}")
    if extra:
        raise PromptRenderError(f"传入了模板里没有的变量: {sorted(extra)}")
    text = _VAR.sub(lambda m: variables[m.group(1)], template)
    if "{{" in text or "}}" in text:
        raise PromptRenderError("渲染后仍残留 {{ }}，模板写法不合规则")
    return text.strip()


def fingerprint(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()
