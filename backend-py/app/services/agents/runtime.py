"""S5 ③ Agent 运行时（移植自 ``agents/index.ts`` 745 行）。

把「一个 Agent 跑一次」拆成三层，便于离线自检：

1. **纯逻辑**（无 IO）：``classify_llm_error`` / ``backoff_delay`` / ``extract_token_usage`` /
   ``normalize_tool_nick`` / ``normalize_tool_output`` / ``assemble_instructions`` / ``model_candidates``；
2. **装配**（读库）：``build_agent_config``（DB 配置 + 工具注册表 + 文本配置）与
   ``append_style_profile``（风格 Profile / 视觉图谱注入）；
3. **驱动**（网络）：``run_agent_with_instructions`` 的「模型 fallback + 瞬态退避重试」循环，
   ``generate`` 与 ``sleep`` **可注入** ⇒ 测试用假驱动，不打真实网络。

⚠️ 与 Node 的**已知且刻意**的差异（都是「依赖未迁」而非取舍）：

* **``rhythm_phase`` 未迁** ⇒ ``storyboard_breaker`` 少一段跨集节奏引导；
* **``subagent`` 未迁** ⇒ ``orchestrator`` 的工具集为空（``list_available_agents`` 也不在）；
* （``DEFAULT_PROMPTS`` **已落地** —— 见 ``services/agent_prompts.py``，出厂提示词现在是真回落值；
  ``skills`` **已落地** —— 见 ``agents/skills.py``，skill 段现在真的会注入。）

⚠️ 传输层是**自建**的：Node 走 Mastra ``agent.generate``（AI SDK 负责工具循环），
而 Python 的文本适配器（``text_adapters``）**不含 tools 支持**（全仓 ``tools`` 出现 0 次）
⇒ 这里用同一个 ``fetch_with_retry`` + 文本适配器的 ``build_request``（保证 URL/鉴权/请求体
形状与文本链路一致），再往 body 里注入 ``tools``，自己跑「工具调用 → 结果回灌」的循环
（``maxSteps`` 上限、``modelSettings`` 的 ``maxOutputTokens``/``temperature`` 都要**显式下发**，
否则 DB 里的这两项是死配置 —— 原 TS 的注释专门写了这个坑）。
**Gemini 的函数调用循环尚未支持**（``:generateContent`` 的 parts 形状不同）⇒ 显式报错，不静默降级。
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from sqlalchemy import select
from sqlalchemy.engine import Connection

from ...models import agent_configs, dramas
from ..adapters.registry import get_text_adapter
from ..agent_prompts import get_default_instructions
from ..agent_registry import VALID_AGENT_TYPES, get_default_name
from ..ai_configs import is_local_config
from ..ai_providers import get_text_config, get_text_provider_base_url
from ..style_profiles import get_active_profile_for_drama
from ..task_logger import (
    log_task_error,
    log_task_progress,
    log_task_success,
    log_task_warn,
    start_trace,
)
from ..vendor_errors import fetch_with_retry, format_vendor_http_error
from ..visual_graph import build_visual_graph_guidance
from .mcp import discover_mcp_tools
from .protocol import build_protocol_contract, parse_agent_protocol
from .skills import load_agent_skills, resolve_default_skills
from .tool import ToolRegistry
from .tools.corpus_tools import create_corpus_tools
from .tools.extract_tools import create_extract_tools
from .tools.grid_prompt_tools import create_grid_prompt_tools
from .tools.script_tools import create_script_tools
from .tools.storyboard_tools import create_storyboard_tools
from .tools.voice_tools import create_voice_tools

__all__ = [
    "AgentRunResult",
    "TokenUsage",
    "append_style_profile",
    "assemble_instructions",
    "backoff_delay",
    "build_agent_config",
    "classify_llm_error",
    "create_agent_tools",
    "extract_token_usage",
    "model_candidates",
    "normalize_tool_nick",
    "normalize_tool_output",
    "run_agent_with_instructions",
    "run_agent_with_retry",
]

#: 单模型内的**瞬态**重试上限（耗尽后换下一个模型）
MAX_TRANSIENT_RETRIES = 3

#: 单次 run 的最大工具步数（Node 默认 20）
DEFAULT_MAX_STEPS = 20

#: ``modelSettings`` 的两个兜底值（原 TS 注释：不显式下发会落到服务端默认，
#: 长回复被截断时**排在文本之后的 tool call 整段丢失**）
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7

#: 会被注入风格 Profile 的 Agent（其余直接透传）
STYLE_PROFILE_TYPES = ("storyboard_breaker", "extractor", "script_rewriter")

#: ``append_style_profile`` 追加的分节标题（**逐字与 TS 一致**）
STYLE_PROFILE_HEADER = "\n\n【项目风格 Profile（house style，最高优先级，必须遵守）】"

#: 只有**空串**要丢掉（``filter(Boolean)``）；``None`` 同样丢
_SKIP = (None, "")


# ===========================================================================
# 纯逻辑
# ===========================================================================


def classify_llm_error(err: BaseException) -> str:
    """失败分类：``"transient"``（可退避重试、耗尽后换模型）/ ``"fatal"``（立即失败）。

    对齐 PenguinHarness 的 StopReason 归一化：认证/参数/超限/审核拦截 ⇒ fatal；
    限流/过载/配额/超时/网络抖动 ⇒ transient。⚠️ **默认 transient**（宁重试，
    避免分类不全误伤可用模型）—— 与原 TS 一致。
    """
    message = f"{getattr(err, 'message', '') or getattr(err, 'args', [''])[0] or ''} " \
              f"{getattr(err, 'name', '') or ''} {getattr(err, 'code', '') or ''}".lower()
    status = (getattr(err, "status", None)
              or getattr(err, "status_code", None)
              or getattr(getattr(err, "response", None), "status", None)
              or getattr(getattr(err, "response", None), "status_code", None))

    if status in (401, 403, 400):
        return "fatal"
    if any(k in message for k in (
        "context length", "context_length", "maximum context", "max tokens", "too many tokens",
        "context window", "token limit", "exceed",
    )):
        return "fatal"
    if any(k in message for k in (
        "content filter", "content_policy", "contentpolicy", "safety", "recitation",
        "moderation", "sensitive", "unsafe",
    )):
        return "fatal"

    if status in (429, 500, 502, 503, 504):
        return "transient"
    if any(k in message for k in (
        "rate limit", "rate_limit", "ratelimit", "too many requests",
    )):
        return "transient"
    if any(k in message for k in ("overloaded", "busy", "capacity", "insufficient_quota", "quota")):
        return "transient"
    if any(k in message for k in (
        "timeout", "timed out", "econnreset", "etimedout", "enotfound", "econnrefused",
        "network", "socket", "aborted", "fetch failed", "connection",
    )):
        return "transient"

    return "transient"


def backoff_delay(retry_index: int) -> int:
    """指数退避：2s 起、×2、**30s 上限**（毫秒）。"""
    return min(2000 * 2 ** retry_index, 30000)


@dataclass
class TokenUsage:
    """单次 Agent 调用的 token 用量。"""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


def extract_token_usage(result: Any) -> TokenUsage | None:
    """从结果里提取 token 用量：**优先累加所有 step**，回退到最后一步的 usage。

    兼容 AI SDK v5（``inputTokens``/``outputTokens``）与 v4（``promptTokens``/
    ``completionTokens``）两套字段名 —— 与原 TS 一致。
    """
    if not isinstance(result, dict):
        return None
    usages: list[dict[str, Any]] = []
    if isinstance(result.get("steps"), list):
        usages = [step["usage"] for step in result["steps"]
                  if isinstance(step, dict) and isinstance(step.get("usage"), dict)]
    if not usages and isinstance(result.get("usage"), dict):
        usages = [result["usage"]]
    if not usages:
        return None

    total = TokenUsage()
    for usage in usages:
        # ⚠️ nullish（`??`）：0 要保留 0，不能用 `or`
        inp = usage.get("inputTokens")
        if inp is None:
            inp = usage.get("promptTokens")
        if inp is None:
            inp = 0
        out = usage.get("outputTokens")
        if out is None:
            out = usage.get("completionTokens")
        if out is None:
            out = 0
        step_total = usage.get("totalTokens")
        if step_total is None:
            step_total = inp + out
        total.input_tokens += inp
        total.output_tokens += out
        total.total_tokens += step_total
    return total


def normalize_tool_nick(tool_call: Any) -> str | None:
    """工具名归一（Mastra 的元素是 ``{type, payload:{toolName,args,result}}``）。"""
    payload = tool_call.get("payload") if isinstance(tool_call, dict) else None
    if isinstance(payload, dict) and payload.get("toolName"):
        return payload["toolName"]
    if not isinstance(tool_call, dict):
        return None
    tool = tool_call.get("tool") if isinstance(tool_call.get("tool"), dict) else {}
    return (tool_call.get("toolName") or tool.get("toolName") or tool.get("id")
            or tool_call.get("name") or tool_call.get("type"))


def normalize_tool_output(tool_result: Any) -> str:
    """工具结果归一：字符串原样，其余 **JSON 紧凑序列化**（``JSON.stringify``）。

    ⚠️ 与 TS 的**入参口径不同**：Node 拿到的永远是 Mastra 的结果信封
    （``{payload:{toolName,args,result}}``），而 ``default_generate`` 直接把**工具的返回值**
    递进来 ⇒ 顶层字符串要**原样返回**（否则会被 ``json.dumps`` 多套一层引号：
    ``abc`` 变成 ``"abc"``）。信封形状的分支保留，行为与 TS 一致。
    """
    if isinstance(tool_result, str):
        return tool_result
    out = None
    if isinstance(tool_result, dict):
        payload = tool_result.get("payload")
        if isinstance(payload, dict) and "result" in payload:
            out = payload["result"]
        else:
            out = tool_result.get("result", tool_result.get("output", tool_result.get("data")))
    if out is None:
        return "null" if tool_result is None else json.dumps(
            tool_result, ensure_ascii=False, separators=(",", ":"))
    if isinstance(out, str):
        return out
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


def assemble_instructions(base_instructions: str, skill_instructions: str | None) -> str:
    """统一组装：``base + skill + 协议契约``（单点，避免重复注入）。"""
    return "\n\n".join(
        part for part in (base_instructions, skill_instructions, build_protocol_contract())
        if part not in _SKIP
    )


def model_candidates(text_config: dict[str, Any], db_model: str | None) -> list[str]:
    """模型候选：``textConfig.models`` 为基础，**DB 指定的模型前置**（去重保序）。"""
    base = list(text_config.get("models") or []) or [text_config.get("model")]
    if not db_model:
        return [m for m in base if m]
    return [db_model, *[m for m in base if m != db_model]]


# ===========================================================================
# 装配（读库）
# ===========================================================================


@dataclass
class AgentConfig:
    """``buildAgentConfig`` 的产物。"""

    text_config: dict[str, Any]
    db_config: Any
    base_instructions: str
    instructions: str
    name: str
    tools: ToolRegistry
    resolved_base_url: str
    #: 保留字段：Node 侧是 ``defaults``（内置提示词对象）；Python 未搬提示词 ⇒ 恒为 None
    defaults: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


def get_agent_config(conn: Connection, agent_type: str) -> Any:
    """取该 Agent 的生效配置：**激活的优先**，否则取第一条（都没则 None）。"""
    rows = conn.execute(
        select(agent_configs).where(agent_configs.c.agent_type == agent_type,
                                   agent_configs.c.deleted_at.is_(None))
    ).all()
    for row in rows:
        if row.is_active:
            return row
    return rows[0] if rows else None


def _as_registry(value: Any) -> ToolRegistry:
    """工具工厂产物（``dict`` / ``ToolRegistry``）统一成 ``ToolRegistry``。"""
    if isinstance(value, ToolRegistry):
        return value
    return ToolRegistry(dict(value or {}))


def create_agent_tools(type: str, episode_id: int, drama_id: int) -> ToolRegistry | None:
    """按 Agent 类型装配工具集（未知类型返回 None ⇒ 该类型不可运行）。

    ⚠️ ``orchestrator`` 在 Node 侧装的是 ``run_subagent`` + ``list_available_agents``，
    而 ``subagent.ts`` **未迁** ⇒ 这里返回**空注册表**（占位）：Agent 类型本身仍存在
    （``validAgentTypes`` 含它、显示名/阶段表都对得上），但它现在**没有可调工具**。
    """
    if type == "script_rewriter":
        return _as_registry(create_script_tools(episode_id))
    if type == "extractor":
        return _as_registry(create_extract_tools(episode_id, drama_id))
    if type == "storyboard_breaker":
        # 这两个 Agent 需要「找相似镜头」的参考 ⇒ 额外挂语料检索工具
        # （⚠️ 语料目录已 gitignore，缺失时工具自身降级返回 available:false）
        return ToolRegistry.of(
            _as_registry(create_storyboard_tools(episode_id, drama_id)),
            _as_registry(create_corpus_tools()),
        )
    if type == "voice_assigner":
        return _as_registry(create_voice_tools(episode_id, drama_id))
    if type == "grid_prompt_generator":
        return ToolRegistry.of(
            _as_registry(create_grid_prompt_tools(episode_id, drama_id)),
            _as_registry(create_corpus_tools()),
        )
    if type == "orchestrator":
        log_task_warn("AgentFactory", "orchestrator-tools-missing", {
            "reason": "subagent.ts 未迁（run_subagent / list_available_agents 暂缺）",
            "episodeId": episode_id, "dramaId": drama_id,
        })
        return ToolRegistry({})
    return None


def append_style_profile(
    conn: Connection, type: str, episode_id: int, drama_id: int, instructions: str
) -> str:
    """把激活的风格 Profile 作为**最高优先级约束**追加到指令末尾。

    仅对 ``storyboard_breaker`` / ``extractor`` / ``script_rewriter`` 生效，且**整体静默降级**
    （读库失败只 warn，绝不阻断 Agent）。分节顺序：shot_patterns → storytelling →
    audio_captions → preferences → qc_rules →（storyboard_breaker 专属）视觉图谱引导；
    最后用 **``'\\n'``** 连接（不是 ``'\\n\\n'``）。
    """
    if type not in STYLE_PROFILE_TYPES:
        return instructions
    try:
        profile = get_active_profile_for_drama(conn, drama_id)
        if not profile:
            return instructions
        parts: list[str] = [instructions, STYLE_PROFILE_HEADER]
        if profile.get("shotPatterns"):
            parts.append(f"镜头语言偏好（shot_patterns）：\n{profile['shotPatterns']}")
        if profile.get("storytelling"):
            parts.append(f"叙事节奏偏好（storytelling）：\n{profile['storytelling']}")
        if profile.get("audioCaptions"):
            parts.append(f"音效/配乐/字幕偏好（audio_captions）：\n{profile['audioCaptions']}")
        if profile.get("preferences"):
            parts.append(f"用户明确偏好（preferences，最高优先级）：\n{profile['preferences']}")
        if profile.get("qcRules"):
            parts.append(f"验收规则（qc_rules，涉及画面/音频标准时必须遵守）：\n{profile['qcRules']}")

        # 多集节奏相位：⚠️ `rhythm-phase.ts` 未迁 ⇒ 这一段**缺失**（已记录为待补）
        if type == "storyboard_breaker" and episode_id:
            log_task_warn("AgentFactory", "rhythm-guidance-skipped", {
                "episodeId": episode_id, "reason": "rhythm-phase.ts 未迁",
            })

        # 视觉图谱：景别/构图/运镜/灯光引导（用剧的 genre/style 取子图）
        if type == "storyboard_breaker" and drama_id:
            try:
                drama = conn.execute(
                    select(dramas.c.genre, dramas.c.style).where(dramas.c.id == drama_id)
                ).first()
                guidance = build_visual_graph_guidance(
                    drama.genre if drama is not None else None,
                    drama.style if drama is not None else None,
                )
                if guidance:
                    parts.append(guidance)
            except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch 等价
                log_task_warn("AgentFactory", "visual-graph-inject-failed", {
                    "dramaId": drama_id, "type": type, "error": str(err),
                })
        return "\n".join(parts)
    except Exception as err:  # noqa: BLE001
        log_task_warn("AgentFactory", "style-profile-inject-failed", {
            "dramaId": drama_id, "type": type, "error": str(err),
        })
        return instructions


def build_agent_config(
    conn: Connection, type: str, episode_id: int, drama_id: int
) -> AgentConfig | None:
    """装配一次 run 所需的全部材料（无工具/未知类型 ⇒ None）。"""
    tools = create_agent_tools(type, episode_id, drama_id)
    if tools is None:
        return None

    db_config = get_agent_config(conn, type)
    # `dbConfig?.systemPrompt?.trim() || defaults.instructions` —— 取不到 DB 配置就回落到出厂提示词
    base_instructions = (
        str(db_config.system_prompt).strip()
        if db_config is not None and db_config.system_prompt else ""
    ) or get_default_instructions(type)
    if not base_instructions:
        log_task_warn("AgentFactory", "instructions-empty", {
            "agentType": type,
            "reason": "DB system_prompt 与出厂提示词都为空（未知类型才会走到）",
        })

    # skill 段：DB 配置优先，否则 SKILL.md 自描述的默认绑定（唯一入口仍是 assemble_instructions）
    skill_instructions = load_agent_skills(
        type, db_config.skills if db_config is not None else None
    )
    instructions = append_style_profile(
        conn, type, episode_id, drama_id,
        assemble_instructions(base_instructions, skill_instructions),
    )
    name = (db_config.name if db_config is not None and db_config.name else "") \
        or get_default_name(type)
    text_config = get_text_config(conn)
    return AgentConfig(
        text_config=text_config,
        db_config=db_config,
        base_instructions=base_instructions,
        instructions=instructions,
        name=name,
        tools=tools,
        resolved_base_url=get_text_provider_base_url(text_config),
    )


def get_agent_defaults() -> list[dict[str, Any]]:
    """``getAgentDefaults()`` —— Agent「出厂默认配置」清单：**默认提示词 + 默认 Skill 绑定**。

    单一事实来源：提示词 = ``agent_prompts``（守卫与 ``agents/index.ts`` 逐字比对）；
    Skill 绑定 = **各 SKILL.md 的 frontmatter ``agents:``**（由 ``skills.resolve_default_skills``
    解析，代码里**不维护**「谁绑谁」的映射）。

    ⇒ 前端「Agent 配置」页的「恢复默认」与未保存时的回显都走这里（``GET /agent-configs/defaults``），
    前端不再各自维护副本 —— 历史上前端那份副本已与后端漂移（还挂着外部 skill 库、提示词里
    引用了不存在的文件），是本项目「提示词多头维护」的主要来源之一。
    """
    return [
        {
            "agent_type": agent_type,
            "name": get_default_name(agent_type),
            "instructions": get_default_instructions(agent_type),
            "skills": resolve_default_skills(agent_type),
        }
        for agent_type in VALID_AGENT_TYPES
    ]


# ===========================================================================
# 驱动（网络）
# ===========================================================================


async def default_generate(
    *,
    config: dict[str, Any],
    model: str,
    instructions: str,
    tools: ToolRegistry,
    message: str,
    max_steps: int,
    model_settings: dict[str, Any],
) -> dict[str, Any]:
    """OpenAI 兼容的**工具调用循环**（自建传输；见模块 docstring）。

    返回 ``{text, tool_calls, tool_results, steps}``，``steps[].usage`` 供
    ``extract_token_usage`` 累加（与 Mastra 的 ``result.steps`` 同形）。
    """
    adapter = get_text_adapter(str(config.get("provider") or ""))
    if getattr(adapter, "provider", "") != "openai-compatible":
        raise ValueError(
            f"Agent runtime: provider '{config.get('provider')}' 使用 "
            f"{type(adapter).__name__}，其函数调用循环尚未移植（仅支持 OpenAI 兼容）"
        )

    is_local = is_local_config(config)
    timeout_ms = 180_000 if is_local else 60_000
    max_retries = 1 if is_local else 3

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": message},
    ]
    tool_calls_out: list[dict[str, Any]] = []
    tool_results_out: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    text = ""

    for _ in range(max(1, int(max_steps))):
        request = adapter.build_request(config, {
            "model": model,
            "messages": messages,
            "temperature": model_settings.get("temperature"),
            "maxTokens": model_settings.get("maxOutputTokens"),
        })
        if tools.tools:
            request["body"]["tools"] = tools.as_openai_tools()

        response = await fetch_with_retry(
            request["url"],
            {
                "method": request["method"],
                "headers": request["headers"],
                # 发给厂商的请求体：紧凑 JSON（与文本链路一致）
                "body": json.dumps(request["body"], ensure_ascii=False, separators=(",", ":"))
                if request.get("body") is not None else None,
            },
            "text",
            timeout_ms=timeout_ms,
            max_retries=max_retries,
        )
        if not response.is_success:
            raise ValueError(
                format_vendor_http_error(response.status_code, response.text, "text")
            )

        data = response.json()
        if isinstance(data.get("usage"), dict):
            steps.append({"usage": data["usage"]})

        choices = data.get("choices") or []
        assistant = (choices[0].get("message") or {}) if choices else {}
        text = assistant.get("content") or ""
        calls = assistant.get("tool_calls") or []
        if not calls:
            break

        messages.append({"role": "assistant", "content": assistant.get("content"),
                         "tool_calls": calls})
        for call in calls:
            function = call.get("function") or {}
            name = function.get("name")
            raw_args = function.get("arguments")
            try:
                args = json.loads(raw_args) if raw_args else {}
            except (ValueError, TypeError):
                args = {}
            tool = tools.get(str(name)) if name else None
            if tool is None:
                output: Any = {"error": f"Unknown tool: {name}"}
            else:
                try:
                    output = await tool.run(args if isinstance(args, dict) else {})
                except Exception as err:  # noqa: BLE001 —— 工具报错回灌给模型，不中断 run
                    output = {"error": str(err)}
            rendered = normalize_tool_output(output)
            tool_calls_out.append({"toolName": name, "args": args})
            tool_results_out.append({"toolName": name, "result": rendered})
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": rendered})

    return {"text": text, "tool_calls": tool_calls_out, "tool_results": tool_results_out,
            "steps": steps}


@dataclass
class AgentRunResult:
    """``runAgentWithInstructions`` 的返回（字段名对齐 TS 的 ``AgentRunResult``）。"""

    model: str
    text: str
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    protocol: dict[str, Any] | None
    protocol_errors: list[str]
    usage: TokenUsage | None


Generate = Callable[..., Awaitable[dict[str, Any]]]


async def run_agent_with_instructions(
    conn: Connection,
    type: str,
    episode_id: int,
    drama_id: int,
    instructions: str,
    message: str,
    options: dict[str, Any] | None = None,
    *,
    generate: Generate | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    mcp_tools: dict[str, Any] | None = None,
) -> AgentRunResult:
    """用**自定义指令**跑一次 Agent（模型 fallback + 瞬态退避重试）。

    供评测/优化闭环使用：直接传候选提示词（纯 base，不含 skill/协议契约），
    由本函数统一组装 —— 保证评测上下文与真实运行一致，且**不碰 DB 的 agent_configs**。

    ``generate`` / ``sleep`` 可注入（测试用假驱动与假时钟）；``mcp_tools`` 为 MCP 外部工具的
    注入点（``mcp.ts`` 未迁 ⇒ 默认空，合并行为保留）。
    """
    options = options or {}
    built = build_agent_config(conn, type, episode_id, drama_id)
    if built is None:
        raise ValueError(f"Invalid agent type: {type}")

    driver: Generate = generate or default_generate
    sleep_fn = sleep or asyncio.sleep
    trace = start_trace("Agent", f"run-{type}",
                        {"agentType": type, "episodeId": episode_id, "dramaId": drama_id})

    # 统一组装：base + skill + 协议契约（与 build_agent_config 同源，避免重复注入 skill）
    skill_instructions = load_agent_skills(
        type, built.db_config.skills if built.db_config is not None else None
    )
    full_instructions = assemble_instructions(instructions, skill_instructions)

    models = model_candidates(built.text_config, built.db_config.model if built.db_config else None)
    if not models:
        trace.error("no-model", {"agentType": type})
        raise ValueError(f"No model configured for agent type {type}")

    max_steps = options.get("maxSteps") or DEFAULT_MAX_STEPS
    model_settings = {
        "maxOutputTokens": ((built.db_config.max_tokens if built.db_config else None)
                            or DEFAULT_MAX_OUTPUT_TOKENS),
        # ⚠️ nullish：DB 里显式写了 0 也要保留 0
        "temperature": ((built.db_config.temperature if built.db_config is not None
                         and built.db_config.temperature is not None else DEFAULT_TEMPERATURE)),
    }

    # MCP 外部工具接入：懒连接 single-flight 发现（**失败只降级，绝不阻断 Agent run**）
    discovered = mcp_tools if mcp_tools is not None else await discover_mcp_tools()
    merged = built.tools
    if discovered:
        # 合并到内置工具之上（同名以 MCP 为准，与 TS 的 `{...built, ...mcp}` 一致）
        merged = ToolRegistry({**built.tools.tools, **discovered})

    last_error: BaseException | None = None

    for attempt, model_name in enumerate(models):
        trace.progress("model-fallback-attempt", {
            "attempt": attempt + 1, "totalModels": len(models), "model": model_name,
        })

        for retry in range(MAX_TRANSIENT_RETRIES + 1):
            try:
                result = await driver(
                    config=built.text_config, model=model_name,
                    instructions=full_instructions, tools=merged, message=message,
                    max_steps=max_steps, model_settings=model_settings,
                )
                text = result.get("text") or ""
                usage = extract_token_usage(result)
                parsed = parse_agent_protocol(text)
                protocol = parsed.get("protocol")
                errors = list(parsed.get("errors") or [])

                trace.success("completed", {
                    "model": model_name,
                    "toolCalls": len(result.get("tool_calls") or []),
                    "protocolStatus": (protocol or {}).get("status") or "missing",
                    "protocolErrors": len(errors),
                    **({"inputTokens": usage.input_tokens,
                        "outputTokens": usage.output_tokens,
                        "totalTokens": usage.total_tokens} if usage else {}),
                })
                return AgentRunResult(
                    model=model_name,
                    text=text,
                    tool_calls=list(result.get("tool_calls") or []),
                    tool_results=list(result.get("tool_results") or []),
                    protocol=protocol,
                    protocol_errors=errors,
                    usage=usage,
                )
            except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch 等价
                last_error = err
                cls = classify_llm_error(err)

                # 非瞬态：重试/换模型都无意义，立即失败（省时间省费用）
                if cls == "fatal":
                    trace.error("fatal-error", {
                        "attempt": attempt + 1, "model": model_name, "error": str(err),
                    })
                    raise

                # 瞬态：先同模型指数退避，耗尽后换下一个模型
                if retry < MAX_TRANSIENT_RETRIES:
                    delay = backoff_delay(retry)
                    trace.progress("backoff-retry", {
                        "attempt": attempt + 1, "retry": retry + 1, "model": model_name,
                        "delayMs": delay, "error": str(err),
                    })
                    await sleep_fn(delay / 1000)
                    continue

                is_last = attempt == len(models) - 1
                trace.error("all-models-failed" if is_last else "model-fallback-error", {
                    "attempt": attempt + 1, "model": model_name, "error": str(err),
                })
                if is_last:
                    raise
                break

    log_task_error("Agent", "run-failed", {"agentType": type, "error": str(last_error)})
    if last_error is not None:
        raise last_error
    raise ValueError("All models failed")


async def run_agent_with_retry(
    conn: Connection,
    type: str,
    episode_id: int,
    drama_id: int,
    message: str,
    options: dict[str, Any] | None = None,
    *,
    generate: Generate | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> AgentRunResult:
    """按 DB/内置配置跑一次 Agent（真实入口）。

    传**纯 baseInstructions**：skill 与协议契约由 ``run_agent_with_instructions``
    统一组装，避免重复注入。
    """
    built = build_agent_config(conn, type, episode_id, drama_id)
    if built is None:
        raise ValueError(f"Invalid agent type: {type}")
    return await run_agent_with_instructions(
        conn, type, episode_id, drama_id, built.base_instructions, message, options,
        generate=generate, sleep=sleep,
    )
