"""Model Adapter 契约（对照课件 1-2）。

业务层（main / 以后的 Loop）只认这几个类型：
  ModelRequest  → 想让模型干什么
  ModelResult   → 模型回了什么
  ModelAdapter  → 怎么把请求发到具体厂商 API

换 DeepSeek Chat / DeepSeek Responses / OpenAI，都不该改 SYSTEM 文案，
只换实现 generate() 的那个类。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

# 结果形态：普通文本 / 已解析的结构化 JSON / 工具调用 / 拒答
ResultKind = Literal["text", "structured", "tool_calls", "refusal"]


@dataclass(frozen=True)
class ModelCapabilities:
    """这个 Adapter 能力清单（给上层做路由/降级用，本步只声明不消费）。"""

    chat_completions: bool  # 是否走 chat.completions
    responses: bool  # 是否走 responses.create
    # native_schema=厂商原生 JSON Schema；json_mode=只保证 JSON 对象；prompt_only=纯靠提示词
    structured_output: Literal["native_schema", "json_mode", "prompt_only"]
    tool_calling: bool
    supports_temperature: bool
    supports_top_p: bool


@dataclass(frozen=True)
class ModelRequest:
    """统一入参：和厂商无关的「一次生成请求」。

    system / user 是语义角色，不是某家 SDK 的字段名。
    Chat 通路会把它们塞进 messages；Responses 通路会映射成 instructions / input。
    """

    system: str
    user: str
    max_output_tokens: int = 1024
    temperature: float | None = None  # None = 不传，用厂商默认
    top_p: float | None = None
    # 非空时：希望模型按该 JSON Schema 输出（本步可先不测）
    output_schema: dict[str, Any] | None = None


@dataclass
class ModelResult:
    """统一出参：屏蔽 chat.completions 与 responses 返回结构差异。"""

    kind: ResultKind
    text: str | None = None  # 原始文本（展示给用户或再解析）
    data: dict[str, Any] | None = None  # output_schema 时解析后的 dict
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # 后续 FC 再用
    finish_reason: str | None = None  # 如 stop / completed
    input_tokens: int = 0
    output_tokens: int = 0
    request_id: str | None = None  # 厂商侧请求 id，排障用


class ModelAdapter(ABC):
    """所有厂商 Adapter 的抽象基类。

    子类必须：
      1) 声明 name、capabilities
      2) 实现 generate：ModelRequest → ModelResult
    """

    name: str
    capabilities: ModelCapabilities

    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResult:
        raise NotImplementedError
