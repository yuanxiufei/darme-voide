"""Agent 输出协议（移植自 ``backend/src/agents/protocol.ts``，79 行）。

**思路**（借鉴 agent-evaluation 的「纯协议输出」）：让每个 Agent 完成任务后在最终回复里输出
一段结构化 **YAML 收尾块**，下游（前端 / 评测 / 优化 / 管线）据此稳定消费，不再依赖 LLM 自由文本。

**解耦**：协议字段语义由 ``build_protocol_contract()`` 的**契约文本**定义（注入 instructions），
代码只负责解析与校验 —— 不在代码里硬编码字段语义。

⚠️ 错误文案是**英文且逐字**（``empty text`` / ``no yaml fence found`` / …），会被上层透出。
⚠️ ``invalid status: …`` 里区分 **缺键**（``undefined``）与**显式 null**（``null``）——
对齐 JS 的 ``JSON.stringify``（``JSON.stringify(undefined)`` 返回 ``undefined`` 本身）。
"""
from __future__ import annotations

import json
import re
from typing import Any

import yaml

__all__ = ["build_protocol_contract", "parse_agent_protocol"]

#: ``\```ya?ml`` 围栏（紧随的 ``\s*\n?`` 与 TS 同形）
_YAML_FENCE = re.compile(r"```ya?ml\s*\n?([\s\S]*?)```", re.IGNORECASE)

#: 区分「键不存在」与「键存在但值为 null」（JS 的 undefined vs null）
_MISSING = object()


def build_protocol_contract() -> str:
    """追加到 instructions 末尾的输出协议契约（**所有 Agent 类型通用**）。"""
    return "\n".join([
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
    ])


def _stringify_status(value: Any) -> str:
    """``JSON.stringify(status)``：缺键 -> ``undefined``、null -> ``null``、字符串 -> 带引号。

    ⚠️ 用**紧凑分隔符**才是真正对齐 ``JSON.stringify``（它从不输出空格）：
    对字符串/标量虽然无差别，但一旦值是个对象，默认的 ``, ``/``: `` 就会多出空格。
    """
    if value is _MISSING:
        return "undefined"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def parse_agent_protocol(text: str | None) -> dict[str, Any]:
    """从 Agent 最终文本里提取并解析协议块，返回 ``{protocol, errors}``。

    失败时 ``protocol`` 为 None、``errors`` 说明原因（**不抛异常**）。
    """
    if not text:
        return {"protocol": None, "errors": ["empty text"]}

    match = _YAML_FENCE.search(text)
    if match is None:
        return {"protocol": None, "errors": ["no yaml fence found"]}

    try:
        parsed = yaml.safe_load(match.group(1))
    except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch 等价
        return {"protocol": None, "errors": [f"yaml parse error: {err}"]}

    if not isinstance(parsed, dict):
        return {"protocol": None, "errors": ["protocol is not an object"]}

    status = parsed.get("status", _MISSING)
    summary = parsed.get("summary", _MISSING)

    if status not in ("ok", "failed"):
        return {"protocol": None, "errors": [f"invalid status: {_stringify_status(status)}"]}
    if not isinstance(summary, str) or not summary.strip():
        return {"protocol": None, "errors": ["missing or empty summary"]}

    # `{ ...obj, status, summary }`：保留 Agent 附加的统计字段，status/summary 用校验后的值
    protocol = {**parsed, "status": status, "summary": summary.strip()}
    return {"protocol": protocol, "errors": []}
