"""把本地 prompts/<name>.txt 推到 Langfuse，生成一个新的不可变版本。

  uv run python scripts/sync_prompt.py scenic_presale                 # 只建版本，不发布
  uv run python scripts/sync_prompt.py scenic_presale --label production  # 建版本并发布

本地文件是源头，进 git 评审；Langfuse 负责版本、发布、回滚。
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from scenic_cs_agent.prompts.registry import _client, local_template
from scenic_cs_agent.prompts.render import render_template, _VAR


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("-m", "--message", default=None)
    args = ap.parse_args()

    text = local_template(args.name)
    # 发布前先按唯一规则渲染一遍，不合规直接拒绝
    render_template(text, {v: "x" for v in _VAR.findall(text)})

    p = _client().create_prompt(
        name=args.name, prompt=text, labels=args.label, type="text",
        commit_message=args.message,
    )
    _client().flush()
    print(f"pushed {args.name} v{p.version} labels={args.label or '(none)'}")


if __name__ == "__main__":
    main()
