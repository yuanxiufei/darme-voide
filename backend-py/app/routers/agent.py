"""Agent 域 —— 与 ``backend/src/routes/agent.ts``（57 行）对齐。

**只有 2 个端点，且都是非流式**（流式在 ``auto-pipeline.ts``，不属于本域）：

* ``POST /api/v1/agent/{type}/chat``  跑一次 Agent（多模型 fallback），返回终态结果；
* ``GET  /api/v1/agent/{type}/debug`` 探活 + 类型合法性。

⚠️ 五处保真点：

1. **两处非法类型的文案不同**（别统一）：``chat`` 是 ``Invalid agent type: {type}``
   （**带类型名**），``debug`` 是 ``Invalid agent type``（**不带**）；
2. **校验顺序**：先查类型 → 再读 body → 打 ``start`` 日志 → **再**校验 ``drama_id``/``episode_id``
   ⇒ 缺 id 时 start 日志**已经打过了**（排查时要能看到那次失败请求的入参）；
3. 缺 id 用 **falsy** 判定（``!episode_id``）⇒ ``0`` 也算缺 ✓；
4. **任何运行期异常都返回 400**（``badRequest``，**不是 500**），文案取 ``str(err)``，
   空则兜底 ``Agent execution failed``；
5. 成功回执固定 ``type: "done"``，``usage`` 是 **camelCase**（``inputTokens``/``outputTokens``/
   ``totalTokens``）或 **null**（上游没给用量时）。

``maxSteps`` 固定 **20**（本路由不透传该参数）。
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request
from sqlalchemy.engine import Connection

from ..db import get_conn
from ..request_utils import read_json
from ..response import bad_request, js_truthy, success
from ..services.agent_registry import VALID_AGENT_TYPES
from ..services.agents.runtime import run_agent_with_retry
from ..services.task_logger import (
    log_task_error,
    log_task_payload,
    log_task_progress,
    log_task_start,
    log_task_success,
)

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])

#: 本路由固定下发的步数上限（原 TS 硬编码 20）
MAX_STEPS = 20


@router.post("/{agent_type}/chat")
async def agent_chat(
    agent_type: str, request: Request, conn: Connection = Depends(get_conn)
):
    """非流式 Agent 对话（带多模型自动 fallback）。"""
    # ⚠️ chat 的文案**带类型名**（debug 不带）
    if agent_type not in VALID_AGENT_TYPES:
        return bad_request(f"Invalid agent type: {agent_type}")

    body = await read_json(request)
    message = body.get("message")
    drama_id = body.get("drama_id")
    episode_id = body.get("episode_id")

    log_task_start("Agent", agent_type, {
        "dramaId": drama_id, "episodeId": episode_id, "message": message,
    })
    log_task_payload("Agent", f"{agent_type} input", body)

    # ⚠️ falsy 判定：0 也算缺；此处 start 日志已经打过（保真，别把校验提前）
    if not js_truthy(episode_id) or not js_truthy(drama_id):
        log_task_error("Agent", agent_type, {"reason": "missing drama_id or episode_id"})
        return bad_request("drama_id and episode_id are required")

    started = _now_ms()
    try:
        result = await run_agent_with_retry(
            conn, agent_type, episode_id, drama_id,
            message if isinstance(message, str) else "",
            {"maxSteps": MAX_STEPS},
        )
        elapsed = _elapsed_seconds(started)
        log_task_success("Agent", agent_type, {"elapsedSeconds": elapsed})
        log_task_progress("Agent", "tool-summary", {
            "agentType": agent_type,
            "toolCalls": [call.get("toolName") for call in result.tool_calls],
            "toolResults": [item.get("toolName") for item in result.tool_results],
        })
        log_task_payload("Agent", f"{agent_type} tool-results", result.tool_results)

        usage = None
        if result.usage is not None:
            usage = {
                "inputTokens": result.usage.input_tokens,
                "outputTokens": result.usage.output_tokens,
                "totalTokens": result.usage.total_tokens,
            }
        return success({
            "type": "done",
            "text": result.text,
            "toolCalls": result.tool_calls,
            "toolResults": result.tool_results,
            "usage": usage,
        })
    except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch 等价：**统一 400**
        elapsed = _elapsed_seconds(started)
        log_task_error("Agent", agent_type, {"elapsedSeconds": elapsed, "error": str(err)})
        return bad_request(str(err) or "Agent execution failed")


@router.get("/{agent_type}/debug")
def agent_debug(agent_type: str):
    """探活：类型合法则 ``valid: true``（⚠️ 文案**不带**类型名）。"""
    if agent_type not in VALID_AGENT_TYPES:
        return bad_request("Invalid agent type")
    return success({"agent_type": agent_type, "valid": True})


def _now_ms() -> float:
    """``performance.now()`` 的等价物（单调时钟，毫秒）。"""
    return time.perf_counter() * 1000


def _elapsed_seconds(started_ms: float) -> str:
    """``((now-start)/1000).toFixed(1)`` —— **字符串**，一位小数。"""
    return f"{(_now_ms() - started_ms) / 1000:.1f}"
