"""景区售前 system prompt：条件逻辑在这里用 Python 拼，模板只放占位。"""

from __future__ import annotations

from scenic_cs_agent.prompts.registry import PromptRegistry, get_prompt
from scenic_cs_agent.prompts.render import RenderedPrompt

PROMPT_NAME = "scenic_presale"


def build_tools_section(allowed_tools: tuple[str, ...]) -> str:
    if allowed_tools:
        return f"已挂载工具：{'、'.join(allowed_tools)}。需要事实数据时必须先调工具。"
    return "当前还没有工具。若用户问具体票价 / 开放时间，明确说需要先查景区资料，并追问景区名。"


def render_scenic_presale(
    *,
    allowed_tools: tuple[str, ...] = (),
    label: str | None = None,
    registry: PromptRegistry | None = None,
) -> RenderedPrompt:
    """渲染售前 system prompt。

    label 默认 None：跟随当前环境配置（test 环境拉 test，生产拉 production）。
    registry 由进程启动时传入。不传时走 get_prompt，给脚本单独使用。
    """

    variables = {"tools_section": build_tools_section(allowed_tools)}
    if registry is not None:
        return registry.get(PROMPT_NAME, variables, label=label)
    return get_prompt(PROMPT_NAME, variables, label=label)
