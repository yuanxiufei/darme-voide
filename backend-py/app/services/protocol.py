"""Agent 输出协议 —— 移植 ``agents/protocol.ts``（整文件）。

借鉴 PenguinHarness agent-evaluation 的「纯协议输出」思想：每个 Agent 完成任务后在最终回复里
输出结构化 YAML 协议块（``status`` / ``summary``），下游（前端 / 评测 / 优化 / 管线）据此
稳定消费，不再依赖 LLM 自由文本。

**协议契约由 instructions 注入**（``build_protocol_contract``），代码只负责解析与校验，
二者解耦 —— 协议字段语义由契约文本定义，不在代码里硬编码。
"""

from __future__ import annotations

import json
import re
from typing import Any

import yaml

#: 提取 YAML fence 块（```yaml 或 ```yml，大小写不敏感）
_FENCE_RE = re.compile(r"```ya?ml\s*\n?(.*?)```", re.S | re.I)


def build_protocol_contract() -> str:
    """追加到 instructions 末尾的输出协议契约（所有 Agent 类型通用）。"""
    return "\n".join(
        [
            "## 输出协议（必须遵守）",
            "",
            "完成任务后，你必须在最后一条回复中输出一个 YAML 代码块作为收尾协议，格式如下：",
            "",
            "```yaml",
            "status: ok        # ok 表示成功完成，failed 表示失败",
            "summary: 一句话概括完成情况和关键数字",
            "```",
            "",
            "要求：",
            "- status 只能填 ok 或 failed",
            "- summary 用一句话概括（例如：已保存 12 个分镜）",
            "- 即使中途工具调用失败，也要输出协议并如实填写 status: failed",
        ]
    )


def parse_agent_protocol(text: str) -> dict[str, Any]:
    """从 Agent 最终文本中提取并解析协议块。

    返回 ``{"protocol": dict | None, "errors": list[str]}``。**不抛错** ——
    协议解析失败是常态（模型没按格式输出），调用方据此记一条 warn 并继续。
    """
    if not text:
        return {"protocol": None, "errors": ["empty text"]}

    match = _FENCE_RE.search(text)
    if not match:
        return {"protocol": None, "errors": ["no yaml fence found"]}

    try:
        parsed = yaml.safe_load(match.group(1))
    except Exception as exc:  # noqa: BLE001
        return {"protocol": None, "errors": [f"yaml parse error: {exc}"]}

    if not isinstance(parsed, dict):
        return {"protocol": None, "errors": ["protocol is not an object"]}

    status = parsed.get("status")
    summary = parsed.get("summary")

    if status not in ("ok", "failed"):
        return {
            "protocol": None,
            "errors": [
                # ⚠️ 紧凑分隔符（Node 是 `JSON.stringify(status)`），这段会进用户可见的报错
                f"invalid status: {json.dumps(status, ensure_ascii=False, separators=(',', ':'))}"
            ],
        }
    if not isinstance(summary, str) or not summary.strip():
        return {"protocol": None, "errors": ["missing or empty summary"]}

    return {"protocol": {**parsed, "status": status, "summary": summary.strip()}, "errors": []}
