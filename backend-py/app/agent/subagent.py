"""子 Agent 调度 —— 移植 ``backend/src/agents/subagent.ts``（107 行，**整域关闭**）。

给主控 Agent（``orchestrator``）装两个工具，实现「Agent 调用 Agent」的递归委托：

* ``run_subagent`` —— 把子任务委托给领域专家 Agent，回传其 protocol/summary/text；
* ``list_available_agents`` —— 列出可委托的专家（**排除自己**）。

硬约束（与 TS 逐条对齐）：

1. ``agent_type`` 必须 ∈ ``VALID_AGENT_TYPES``（原 TS 是运行时动态 import 校验）；
2. **禁止自调用**（``agent_type == parent_type``）；
3. ``MAX_SUBAGENT_DEPTH = 2`` 深度上限（防无限递归）；
4. **环形调用检测**（A→B→A）——TS 用 ``AsyncLocalStorage`` 追踪调用链，Python 侧用
   :class:`contextvars.ContextVar`（同样跨 ``await`` 传播，且**天然按任务隔离**）。

⚠️ 实际上只有 ``orchestrator`` 会被装上这两个工具（见 ``runtime.create_agent_tools``），
而领域 Agent 没有 ``run_subagent`` ⇒ 递归天然只走一层；上面 3/4 两条是**纵深防御**（对齐原实现）。

⚠️ 四个「错误结果」都是**返回值**（``{"error": ...}``）而不是抛异常 —— 工具错误要回到模型手里
让它自己改主意，抛异常会中断整个 run（原 TS 同样如此）。
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from sqlalchemy.engine import Connection

from app.agent.tool import Tool, json_string, object_schema

__all__ = ["MAX_SUBAGENT_DEPTH", "SUBAGENT_REGISTRY", "create_run_subagent_tools"]

#: 领域专家 Agent 能力清单（供 orchestrator 决策调度）—— **逐字**对齐 TS
SUBAGENT_REGISTRY: tuple[dict[str, str], ...] = (
    {"type": "script_rewriter", "name": "剧本改写",
     "capability": "将小说/原始内容改写为格式化短剧剧本并保存到当前集"},
    {"type": "extractor", "name": "角色场景提取",
     "capability": "从剧本提取角色与场景（同名/同地点智能去重）并保存"},
    {"type": "storyboard_breaker", "name": "分镜拆解",
     "capability": "将剧本拆解为带完整字段的分镜方案并保存"},
    {"type": "voice_assigner", "name": "角色音色分配",
     "capability": "为每个角色匹配最合适的音色并保存"},
    {"type": "grid_prompt_generator", "name": "图片提示词生成",
     "capability": "生成角色/场景/宫格图的英文提示词"},
)

#: 最大委托深度：orchestrator → 领域 Agent（领域 Agent 无 run_subagent，天然终止）
MAX_SUBAGENT_DEPTH = 2

#: 调用链（对齐 TS 的 ``AsyncLocalStorage<SubagentChainContext>``）
_chain: ContextVar[tuple[str, ...] | None] = ContextVar("subagent_chain", default=None)


def current_chain() -> tuple[str, ...] | None:
    """当前调用链（仅供自检观察，业务不读）。"""
    return _chain.get()


def create_run_subagent_tools(conn: Connection, drama_id: int, episode_id: int,
                             parent_type: str) -> dict[str, Tool]:
    """创建子 Agent 调度工具集（``run_subagent`` + ``list_available_agents``）。

    ``conn`` 用于把委托跑成一次真实的 Agent run（``run_agent_with_retry``）。
    """
    async def _run_subagent(arguments: dict[str, Any]) -> dict[str, Any]:
        agent_type = str(arguments.get("agent_type") or "")
        task = str(arguments.get("task") or "")

        # 惰性导入：runtime ↔ subagent 互相引用（TS 侧同样用动态 import 打破）
        from app.services.agent_registry import VALID_AGENT_TYPES  # noqa: PLC0415
        from app.agent.runtime import run_agent_with_retry  # noqa: PLC0415

        if agent_type not in VALID_AGENT_TYPES:
            return {"error": f"Unknown agent type: {agent_type}. "
                             f"Available: {', '.join(VALID_AGENT_TYPES)}"}
        if agent_type == parent_type:
            return {"error": f"Cannot delegate to self ({agent_type})"}

        chain = _chain.get() or (parent_type,)
        if agent_type in chain:
            return {"error": "Circular delegation detected: "
                             + " -> ".join([*chain, agent_type])}
        if len(chain) >= MAX_SUBAGENT_DEPTH:
            return {"error": f"Max subagent depth reached ({MAX_SUBAGENT_DEPTH})"}

        token = _chain.set((*chain, agent_type))
        try:
            result = await run_agent_with_retry(
                conn, agent_type, episode_id, drama_id, task)
        except Exception as exc:  # noqa: BLE001 —— 与原 TS 的 catch 等价
            return {"error": f"Subagent {agent_type} failed: {exc}"}
        finally:
            # ⚠️ 无论成败都要还原链 —— 否则同一次 run 里后续委托会带着已走过的链
            _chain.reset(token)

        protocol = result.protocol or {}
        return {
            "status": protocol.get("status") or "completed",
            "summary": protocol.get("summary") or "",
            "text": result.text,
            "tool_calls_count": len(result.tool_calls),
        }

    async def _list_available_agents(_arguments: dict[str, Any]) -> dict[str, Any]:
        return {"agents": [entry for entry in SUBAGENT_REGISTRY
                           if entry["type"] != parent_type]}

    return {
        "run_subagent": Tool(
            id="run_subagent",
            description="Delegate a subtask to a specialist agent and return its result. "
                        "Use this to break a complex job into domain-expert tasks "
                        "executed in order.",
            input_schema=object_schema({
                "agent_type": json_string(
                    "Target specialist agent type (see list_available_agents)"),
                "task": json_string(
                    "The concrete instruction for the subagent, with enough context"),
            }),
            execute=_run_subagent,
        ),
        "list_available_agents": Tool(
            id="list_available_agents",
            description="List the specialist agents available for delegation "
                        "and their capabilities.",
            # `z.object({})` ⇒ 空属性、无必填
            input_schema=object_schema({}, required=[]),
            execute=_list_available_agents,
        ),
    }
