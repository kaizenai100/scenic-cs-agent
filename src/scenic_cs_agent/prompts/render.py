"""Prompt 模板渲染。

要点：
- Jinja2 + StrictUndefined：模板里用了未传入的变量会直接报错，避免静默空字符串
- 模板文件带版本号（文件名 / prompt_version），方便以后对照评测
- fingerprint：渲染结果的 sha256，排障时知道「当时到底喂了哪段 system」
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from jinja2 import Environment, StrictUndefined, select_autoescape

PROMPTS_DIR = Path(__file__).resolve().parent

# autoescape 默认关：这是给模型的文本，不是 HTML 页面
_env = Environment(
    undefined=StrictUndefined,
    autoescape=select_autoescape(default=False),
    trim_blocks=True,
    lstrip_blocks=True,
)


@dataclass(frozen=True)
class RenderedPrompt:
    """一次渲染结果：正文 + 版本 + 指纹。"""

    text: str
    version: str
    fingerprint: str
    template_name: str


def render_scenic_presale(
    *,
    tools_enabled: bool = False,
    allowed_tools: tuple[str, ...] = (),
    prompt_version: str = "v1",
) -> RenderedPrompt:
    """渲染景区售前 system prompt。

    tools_enabled=False：步骤 4 默认，和步骤 1–3 行为对齐。
    以后接上 Mock 工具时再打开。
    """
    template_name = f"scenic_presale_{prompt_version}.md.j2"
    path = PROMPTS_DIR / template_name
    if not path.exists():
        raise FileNotFoundError(f"prompt template not found: {path}")

    template = _env.from_string(path.read_text(encoding="utf-8"))
    text = template.render(
        tools_enabled=tools_enabled,
        allowed_tools=list(allowed_tools),
        prompt_version=prompt_version,
    ).strip()
    fingerprint = sha256(text.encode("utf-8")).hexdigest()
    return RenderedPrompt(
        text=text,
        version=prompt_version,
        fingerprint=fingerprint,
        template_name=template_name,
    )
