"""结构化输出执行器：调用 → 解析 → 校验 → 带错重试 → 兜底。

链路：
  1. adapter.generate(output_schema=...)，DeepSeek 用 json_object 模式
  2. json.loads + Pydantic 校验
  3. 失败：把「上次输出 + 报错」追加进 user，让模型自己修，最多 max_repairs 次
  4. 还失败：返回 fallback（intent=unknown + 安全话术），业务不崩
每次尝试都记一条 attempt，方便排障看到底错在哪。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from pydantic import ValidationError

from scenic_cs_agent.runtime import ModelAdapter, ModelRequest
from scenic_cs_agent.structured.schemas import IntentResult, Slots

FALLBACK_REPLY = "抱歉，我没太理解。您想咨询哪个景区、哪天去、几位游客呢？"


@dataclass
class Attempt:
    index: int
    raw_text: str | None
    error: str | None


@dataclass
class StructuredOutcome:
    result: IntentResult
    used_fallback: bool
    attempts: list[Attempt] = field(default_factory=list)


def _fallback() -> IntentResult:
    return IntentResult(
        intent="unknown",
        confidence=0.0,
        slots=Slots(),
        missing_slots=["scenic_name", "visit_date", "visitors"],
        reply=FALLBACK_REPLY,
    )


def classify_intent(
    adapter: ModelAdapter,
    *,
    system: str,
    user: str,
    max_repairs: int = 2,
) -> StructuredOutcome:
    schema = IntentResult.model_json_schema()
    attempts: list[Attempt] = []
    prompt_user = user

    for i in range(max_repairs + 1):
        raw_text: str | None = None
        try:
            res = adapter.generate(
                ModelRequest(
                    system=system,
                    user=prompt_user,
                    output_schema=schema,
                    temperature=0,
                    max_output_tokens=512,
                )
            )
            raw_text = res.text
            # 不信 adapter 里的 data，统一在这里自己解析+校验
            parsed = IntentResult.model_validate(json.loads(raw_text or ""))
            attempts.append(Attempt(i, raw_text, None))
            return StructuredOutcome(parsed, False, attempts)
        except (json.JSONDecodeError, ValidationError) as e:
            err = str(e)
            attempts.append(Attempt(i, raw_text, err))
            # 错误回灌：把上次输出和报错一起给模型
            prompt_user = (
                f"{user}\n\n[上次输出不合法，请修正后只输出 JSON]\n"
                f"上次输出：{raw_text}\n校验错误：{err[:800]}"
            )

    return StructuredOutcome(_fallback(), True, attempts)
