"""S5 ③ Agent 运行时（移植自 ``agents/index.ts`` 745 行）。

把「一个 Agent 跑一次」拆成三层，便于离线自检：

1. **纯逻辑**（无 IO）：``classify_llm_error`` / ``backoff_delay`` / ``extract_token_usage`` /
   ``normalize_tool_nick`` / ``normalize_tool_output`` / ``assemble_instructions`` / ``model_candidates``；
2. **装配**（读库）：``build_agent_config``（DB 配置 + 工具注册表 + 文本配置）与
   ``append_style_profile``（风格 Profile / 视觉图谱注入）；
3. **驱动**（网络）：``run_agent_with_instructions`` 的「模型 fallback + 瞬态退避重试」循环，
   ``generate`` 与 ``sleep`` **可注入** ⇒ 测试用假驱动，不打真实网络。

✅ 曾经的「依赖未迁」项**现已全部落地接线**：
``rhythm_phase``（``services/rhythm_phase.py``）⇒ ``storyboard_breaker`` 注入跨集节奏引导；``subagent``
（``agents/subagent.py``）⇒ ``orchestrator`` 装 ``run_subagent`` + ``list_available_agents``，可自主委托
领域专家（深度 2、防自调用/防环形）；``DEFAULT_PROMPTS``（``services/agent_prompts.py``，出厂提示词是
真回落值）；``skills``（``agents/skills.py``，skill 段真的注入）；MCP 工具发现见下方 ``discover_mcp_tools``。

⚠️ 传输层是**自建**的：Node 走 Mastra ``agent.generate``（AI SDK 负责工具循环），
而 Python 的文本适配器（``text_adapters``）**不含 tools 支持**（全仓 ``tools`` 出现 0 次）
⇒ 这里用同一个 ``fetch_with_retry`` + 文本适配器的 ``build_request``（保证 URL/鉴权/请求体
形状与文本链路一致），再往 body 里注入 ``tools``，自己跑「工具调用 → 结果回灌」的循环
（``maxSteps`` 上限、``modelSettings`` 的 ``maxOutputTokens``/``temperature`` 都要**显式下发**，
否则 DB 里的这两项是死配置 —— 原 TS 的注释专门写了这个坑）。
⭐ **Ollama 原生**（``/api/chat``）的工具循环自 2026-10-28 起已支持 ✓ —— 本机默认文本配置就是
``provider=ollama``，此前 ``/api/v1/agent/{type}/chat`` 恒 400「尚未移植」✗。它的
``message.tool_calls`` 与 OpenAI **同形**，只有两处差异：``function.arguments`` 是**对象**（不是 JSON
字符串）、工具结果用 ``{"role":"tool","tool_name":…}`` 回灌（没有 ``tool_call_id``）。
**Gemini 仍未支持**（``:generateContent`` 的 parts 形状不同）⇒ 显式报错，不静默降级。
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from sqlalchemy import select
from sqlalchemy.engine import Connection

from app.core.models import agent_configs, dramas
from app.services.adapters.registry import get_text_adapter
from app.services.adapters.text_adapters import ollama_num_ctx
from app.services.agent_prompts import get_default_instructions
from app.services.agent_registry import (
    AGENT_SPECS,
    STYLE_PROFILE_TYPES,
    VALID_AGENT_TYPES,
    get_agent_spec,
    get_default_name,
)
from app.services.ai_configs import is_local_config
from app.services.ai_providers import get_text_config, get_text_provider_base_url
from app.services.style_profiles import get_active_profile_for_drama
from app.services.task_logger import (
    log_task_error,
    log_task_progress,
    log_task_success,
    log_task_warn,
    start_trace,
)
from app.services.vendor_errors import fetch_with_retry, format_vendor_http_error
from app.services.visual_graph import build_visual_graph_guidance
from app.services import rhythm_phase
from app.mcp.client import discover_mcp_tools
from app.agent import context_budget
from app.agent.protocol import build_protocol_contract, parse_agent_protocol
from app.agent.skills import load_agent_skills, resolve_default_skills
from app.agent.tool import ToolRegistry
from app.agent.tools.corpus_tools import create_corpus_tools
from app.agent.tools.extract_tools import create_extract_tools
from app.agent.tools.grid_prompt_tools import create_grid_prompt_tools
from app.agent.tools.script_tools import create_script_tools
from app.agent.tools.storyboard_tools import create_storyboard_tools
from app.agent.tools.voice_tools import create_voice_tools

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

#: 会被注入风格 Profile 的 Agent（其余直接透传）。
#: 成员集合的单一来源在 ``agent_registry.AGENT_SPECS`` 的 ``style_profile`` 字段 ——
#: 本文件只 import（不再复述），加 Agent 时改一处即可。

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
    # ⚠️ 上下文超窗是**确定性**失败：重试、退避、换模型都不会变好（同一份消息）
    # ⇒ 必须打 fatal，否则要白等 8 轮退避再报同一个错。用类型判而不是靠字符串匹配。
    if isinstance(err, context_budget.ContextOverflowError):
        return "fatal"
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


#: 工具集标识 → 工厂。**签名统一**为 ``(conn, type, episode_id, drama_id) -> ToolRegistry``。
#:
#: ⚠️ 用「声明 + 查表」而不是 if/elif 分派：漏掉一个分支的症状是**静默装配出 ``None``**
#: ⇒ 该 Agent 报「不可运行」，却看不出是"忘了加分支"。加 Agent 现在只需在
#: ``agent_registry._STRUCTURE`` 声明 ``tool_set`` + 在下表加一条映射。
_TOOL_FACTORIES: dict[str, Callable[..., ToolRegistry]] = {}


def _tool_factory(tool_set: str) -> Callable[..., ToolRegistry | None]:
    """登记一个工具集工厂（装饰器形式，避免「函数名与表中的键」两处各写一遍）。"""
    def register(factory: Callable[..., ToolRegistry]) -> Callable[..., ToolRegistry]:
        _TOOL_FACTORIES[tool_set] = factory
        return factory

    return register


@_tool_factory("script")
def _tools_script(conn: Connection, type: str, episode_id: int,
                  drama_id: int) -> ToolRegistry:
    return _as_registry(create_script_tools(episode_id))


@_tool_factory("extract")
def _tools_extract(conn: Connection, type: str, episode_id: int,
                   drama_id: int) -> ToolRegistry:
    return _as_registry(create_extract_tools(episode_id, drama_id))


@_tool_factory("storyboard")
def _tools_storyboard(conn: Connection, type: str, episode_id: int,
                      drama_id: int) -> ToolRegistry:
    # 这两个 Agent 需要「找相似镜头」的参考 ⇒ 额外挂语料检索工具
    # （⚠️ 语料目录已 gitignore，缺失时工具自身降级返回 available:false）
    return ToolRegistry.of(
        _as_registry(create_storyboard_tools(episode_id, drama_id)),
        _as_registry(create_corpus_tools()),
    )


@_tool_factory("voice")
def _tools_voice(conn: Connection, type: str, episode_id: int,
                 drama_id: int) -> ToolRegistry:
    return _as_registry(create_voice_tools(episode_id, drama_id))


@_tool_factory("grid")
def _tools_grid(conn: Connection, type: str, episode_id: int,
                drama_id: int) -> ToolRegistry:
    return ToolRegistry.of(
        _as_registry(create_grid_prompt_tools(episode_id, drama_id)),
        _as_registry(create_corpus_tools()),
    )


@_tool_factory("subagent")
def _tools_subagent(conn: Connection, type: str, episode_id: int,
                    drama_id: int) -> ToolRegistry:
    from app.agent.subagent import create_run_subagent_tools  # noqa: PLC0415 —— 打破循环导入

    return ToolRegistry(create_run_subagent_tools(
        conn, drama_id, episode_id, parent_type=type))


def create_agent_tools(conn: Connection, type: str, episode_id: int,
                       drama_id: int) -> ToolRegistry | None:
    """按 Agent 类型装配工具集（未知类型 / 未声明工具集 ⇒ ``None`` ⇒ 该类型不可运行）。

    ``orchestrator`` 装的是 ``run_subagent`` + ``list_available_agents``（``agents/subagent.py``，
    2026-09-15 起**已迁**；早先这里是空注册表占位 + 一条 ``orchestrator-tools-missing`` warn）。
    ⇒ 因此本函数需要 ``conn``（子 Agent 委托要跑一次真实 Agent run）。

    分派依据 = ``agent_registry.AGENT_SPECS[*].tool_set``（**单一来源**；本文件不再有 if/elif）。
    """
    spec = get_agent_spec(type)
    factory = _TOOL_FACTORIES.get(spec.tool_set) if spec and spec.tool_set else None
    if factory is None:
        return None
    return factory(conn, type, episode_id, drama_id)


def _assert_tool_sets_registered() -> None:
    """每个声明的 ``tool_set`` 都必须有工厂 —— 否则**导入期硬失败**。

    这是有意的「宁可起不来」：漏注册的症状是 ``create_agent_tools`` 静默返回 ``None``，
    该 Agent 在界面上表现为「不可运行」，排查时会一路怀疑到 DB/权限/路由，而真正的
    原因只是「少写了个装饰器」。
    """
    declared = {spec.tool_set for spec in AGENT_SPECS if spec.tool_set}
    missing = sorted(declared - set(_TOOL_FACTORIES))
    if missing:
        raise RuntimeError(
            f"Agent 声明了工具集但未注册工厂：{missing}。"
            f"请在 runtime 用 @_tool_factory(\"<tool_set>\") 注册。"
        )


_assert_tool_sets_registered()


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

        # 多集节奏相位：⭐ 2026-09-25 接线 ✓ —— 早先这里是「`rhythm-phase.ts` 未迁」的 warn 占位 ✗，
        # 而 `services/rhythm_phase.py` 的 `rhythm_guidance_for_episode` 早就迁好了 ⇒ `storyboard_breaker`
        # 一直少一段跨集节奏引导 ✓✗（与「能力已有、没接出去」同族 ✓）。现在真调它 ✓。
        if type == "storyboard_breaker" and episode_id:
            try:
                guidance = rhythm_phase.rhythm_guidance_for_episode(conn, episode_id)
                if guidance:
                    parts.append(guidance)
            except Exception as err:  # noqa: BLE001 —— 读库失败只 warn 不阻断 ✓（与视觉图谱分支同口径 ✓）
                log_task_warn("AgentFactory", "rhythm-guidance-inject-failed", {
                    "episodeId": episode_id, "type": type, "error": str(err),
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


def assemble_full_instructions(
    conn: Connection, type: str, episode_id: int, drama_id: int,
    base_instructions: str, skill_instructions: str | None,
) -> str:
    """最终 system prompt 的**唯一组装点**：``base → skill → 协议契约 → 风格注入``。

    ⚠️ 为什么必须只有一个（历史缺陷，2026-09-25 修）：
    这里曾经有**两个组装点** —— ``build_agent_config`` 算出了带风格 Profile 的
    ``AgentConfig.instructions``，而真实运行入口 ``run_agent_with_retry`` 传给
    ``run_agent_with_instructions`` 的是**纯 ``base_instructions``**，后者再自行
    组装（只做 base+skill+契约，**不含风格注入**）。两份结果静默分叉 ⇒
    ``append_style_profile`` 追加的风格 Profile / 跨集节奏相位 / 视觉图谱
    **从未进过模型**，全仓唯一的读取者是测试（``skills_test`` / ``agent_prompts_test``）。
    这类缺陷不会报错、不会失败，只表现为「模型不遵守风格约束」——
    所以改为两条路径共用本函数，结构上不可能再分叉。
    """
    return append_style_profile(
        conn, type, episode_id, drama_id,
        assemble_instructions(base_instructions, skill_instructions),
    )


def build_agent_config(
    conn: Connection, type: str, episode_id: int, drama_id: int
) -> AgentConfig | None:
    """装配一次 run 所需的全部材料（无工具/未知类型 ⇒ None）。"""
    tools = create_agent_tools(conn, type, episode_id, drama_id)
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
    instructions = assemble_full_instructions(
        conn, type, episode_id, drama_id, base_instructions, skill_instructions,
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


def _transport_retry(config: dict[str, Any]) -> tuple[int, int]:
    """本机服务的超时/重试口径（与 Node 的 ``isLocal`` 分支一致）。"""
    is_local = is_local_config(config.get("baseUrl") or "", str(config.get("provider") or ""))
    return (180_000 if is_local else 60_000, 1 if is_local else 3)


def _parse_tool_args(raw: Any) -> dict[str, Any]:
    """工具入参归一：OpenAI 回**字符串**、Ollama 回**对象**，两种都要吃得下。"""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _ollama_context_budget(
    config: dict[str, Any], model_settings: dict[str, Any]
) -> context_budget.Budget:
    """Ollama 的**输入侧**预算 ＝ ``num_ctx`` − 输出预留（``num_predict``）。

    ⚠️ 为什么必须扣（2026-10-28 实测）：Ollama 的 ``num_ctx`` 是 **prompt + 输出共用一个窗口**，
    而本项目的 ``maxOutputTokens`` 默认 4096 与 Ollama 的默认窗口 4096 同值 ⇒ 提示词被压到
    只剩 ~2048（实测 ``prompt_eval_count`` 2050）且**丢开头**＝丢 system instructions，不报错。
    窗口已由适配器显式下发（``OLLAMA_DEFAULT_NUM_CTX``），这里把预算阈值也对齐同一个数，
    否则 ``context_budget`` 仍按 24000 估算 ⇒ 算出来的数与本机窗口无关，压缩层形同虚设。

    默认 32768 ⇒ 预算 28672（最能吃的 ``storyboard_breaker`` 固定开销 16276，余 12396）。
    """
    num_ctx = ollama_num_ctx(config)
    try:
        reserved = int(model_settings.get("maxOutputTokens") or 0)
    except (TypeError, ValueError):
        reserved = 0
    reserved = reserved if reserved > 0 else DEFAULT_MAX_OUTPUT_TOKENS
    return context_budget.Budget(limit_tokens=max(1_000, num_ctx - reserved))


def _apply_context_budget(
    messages: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    budget: context_budget.Budget | None = None,
    *,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """上下文预算的**机械层**（两条工具循环共用；零 LLM 调用、消息条数不变）。

    ⚠️ **超预算必须留痕**（2026-09-25 修）：此前只在「有老 tool 输出可压」时才记 step ✗，
    于是「机械层压不动」这种情况（``needsSummarize`` —— 中间没有可压的 tool 输出 /
    最近 ``keep_recent`` 条一律不动）在运行记录里**完全查不到** ✗，
    外部只表现为「模型毫无征兆地开始丢前文」✗。
    本循环仍**刻意不**擅自调模型做摘要（成本 + 失真，见 ``context_budget`` 模块文档）。

    ``strict=True`` 时（Ollama 路径）：机械层压完仍超 ⇒ 抛
    :class:`~app.agent.context_budget.ContextOverflowError`，**不再照发**。
    理由见该异常类的注释：Ollama 超窗是 HTTP 200 + 静默丢开头，
    「发出去拿到一份错的稿」比「当场报错」代价大得多。OpenAI 兼容路径保持只 warn
    （云端窗口通常 ≥32k，且改成本地这种硬失败会误伤）。
    """
    settings = budget or context_budget.Budget()
    budget_plan = context_budget.plan_compression(messages, settings)
    if budget_plan.elide:
        messages = context_budget.apply_plan(messages, budget_plan)
    if budget_plan.overBudget:
        steps.append({"contextBudget": budget_plan.to_dict()})
        if budget_plan.needsSummarize:
            log_task_warn("Agent", "context-over-budget", {
                "estimatedTokens": round(budget_plan.tokensAfter),
                "limitTokens": settings.limit_tokens,
                "messageCount": len(messages),
                "strict": strict,
            })
            if strict:
                raise context_budget.ContextOverflowError(budget_plan, settings.limit_tokens)
    return messages


#: 只读工具前缀 —— 与前端 ``useAgent.ts`` 的 ``READ_ONLY_PREFIXES`` 同一口径。
_READ_ONLY_PREFIXES = ("read_", "list_", "get_")

#: 「读了就收工」时补发的催促（整个 run 只发一次）
_NUDGE_TEXT = (
    "你还没有完成任务：上一轮只读取了数据，没有调用任何保存工具，数据库没有发生任何变化。"
    "请立刻按工作流程继续，调用保存类工具（save_* 等）把结果写入；"
    "不要只输出分析、点评或建议，也不要询问用户是否需要继续。"
    "注意：保存时的数组不能传空（characters 至少要覆盖剧本里所有有名有姓的人物），"
    "空数组会被工具拒绝并返回错误。"
)


def _called_tools(tool_calls_out: list[dict[str, Any]]) -> list[str]:
    return [str(call.get("toolName") or "") for call in tool_calls_out if call.get("toolName")]


def _need_write_nudge(
    tool_calls_out: list[dict[str, Any]],
    text: str,
    already_nudged: bool,
    tools: "ToolRegistry",
) -> bool:
    """模型「读完数据就收工」时，是否需要补一句话催它去保存。

    ⚠️ 为什么需要（2026-10-28 实测）：本机文本服务是 ``qwen3:14b``（本地 Ollama），
    ``extractor`` 只调了 ``read_script_for_extraction`` 就返回一篇「故事亮点分析」，
    一个 ``save_dedup_*`` 都没调 ⇒ 库里一行没变，而接口仍返回 200。
    小模型的 agent 自觉性靠不住，这里由运行时补一轮明确的催促（**不依赖提示词自觉**）。

    只有同时满足才催：有工具可调、确实调过工具、调的全是只读、并且模型已经说了话。
    —— 纯问答（一个工具都没调）不催；已经落过库的不催；同一个 run 催过一次就不再催。
    """
    if already_nudged or not tools.tools or not (text or "").strip():
        return False
    names = _called_tools(tool_calls_out)
    if not names:
        return False
    return all(name.startswith(_READ_ONLY_PREFIXES) for name in names)


async def _run_tool(
    tools: ToolRegistry,
    name: Any,
    args: dict[str, Any],
    tool_calls_out: list[dict[str, Any]],
    tool_results_out: list[dict[str, Any]],
) -> str:
    """跑一个工具并把结果序列化（工具报错**回灌给模型**，不中断 run）。"""
    tool = tools.get(str(name)) if name else None
    if tool is None:
        output: Any = {"error": f"Unknown tool: {name}"}
    else:
        try:
            output = await tool.run(args)
        except Exception as err:  # noqa: BLE001 —— 工具报错回灌给模型，不中断 run
            output = {"error": str(err)}
    rendered = normalize_tool_output(output)
    tool_calls_out.append({"toolName": name, "args": args})
    tool_results_out.append({"toolName": name, "result": rendered})
    return rendered


def _ollama_usage(data: dict[str, Any]) -> dict[str, Any] | None:
    """Ollama 用量（``prompt_eval_count``/``eval_count``）→ ``extract_token_usage`` 的形状。"""
    prompt_eval = data.get("prompt_eval_count")
    eval_count = data.get("eval_count")
    if not isinstance(prompt_eval, int) and not isinstance(eval_count, int):
        return None
    input_tokens = prompt_eval if isinstance(prompt_eval, int) else 0
    output_tokens = eval_count if isinstance(eval_count, int) else 0
    return {"inputTokens": input_tokens, "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens}


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
    """工具调用循环的**入口**（自建传输；见模块 docstring）。

    返回 ``{text, tool_calls, tool_results, steps}``，``steps[].usage`` 供
    ``extract_token_usage`` 累加（与 Mastra 的 ``result.steps`` 同形）。

    ⭐ 2026-10-28：**Ollama 原生**（``/api/chat``）也走这条路 —— 此前只认 OpenAI 兼容，
    而本机默认文本配置正是 ``provider=ollama`` ⇒ ``/agent/{type}/chat`` 恒 400。
    其余原生适配器（Gemini）仍**显式报错**，不静默降级。
    """
    adapter = get_text_adapter(str(config.get("provider") or ""))
    provider = getattr(adapter, "provider", "")
    loop_kwargs: dict[str, Any] = {
        "adapter": adapter,
        "config": config,
        "model": model,
        "instructions": instructions,
        "tools": tools,
        "message": message,
        "max_steps": max_steps,
        "model_settings": model_settings,
    }
    if provider == "ollama":
        return await _run_ollama_tool_loop(**loop_kwargs)
    if provider == "openai-compatible":
        return await _run_openai_tool_loop(**loop_kwargs)
    raise ValueError(
        f"Agent runtime: provider '{config.get('provider')}' 使用 "
        f"{type(adapter).__name__}，其函数调用循环尚未移植（支持 OpenAI 兼容 / Ollama 原生）"
    )


async def _run_openai_tool_loop(
    *,
    adapter: Any,
    config: dict[str, Any],
    model: str,
    instructions: str,
    tools: ToolRegistry,
    message: str,
    max_steps: int,
    model_settings: dict[str, Any],
) -> dict[str, Any]:
    """OpenAI 兼容（``/chat/completions``）的工具调用循环。"""
    timeout_ms, max_retries = _transport_retry(config)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": message},
    ]
    tool_calls_out: list[dict[str, Any]] = []
    tool_results_out: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    text = ""
    nudged = False

    for _ in range(max(1, int(max_steps))):
        # ⚠️ **上下文预算**（移植自 Mini-Agent 的思路 ✓）：
        #    这个循环每步把**整份 messages** 重发 ✓ ⇒ 多步工具链会一路撑大 ✗。
        #    这里做的是**只机械层**（老的 tool 输出换占位正文 ✓）—— 零 LLM 调用 ✓、
        #    **消息条数不变** ✓ ⇒ tool 往返的成对关系不会被破坏 ✓（破坏它会直接 400 ✗）。
        #    机械层不够时会带出 ``needsSummarize`` ✓（**要不要花那次摘要调用由上层决定** ✗ —— 
        #    本循环不擅自调模型 ✓）。
        messages = _apply_context_budget(messages, steps)
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
            if _need_write_nudge(tool_calls_out, text, nudged, tools):
                nudged = True
                steps.append({"nudge": "missing-write"})
                log_task_warn("Agent", "missing-write-nudge", {
                    "calledTools": _called_tools(tool_calls_out), "textChars": len(text),
                })
                messages.append({"role": "user", "content": _NUDGE_TEXT})
                continue
            break

        messages.append({"role": "assistant", "content": assistant.get("content"),
                         "tool_calls": calls})
        for call in calls:
            function = call.get("function") or {}
            name = function.get("name")
            args = _parse_tool_args(function.get("arguments"))
            rendered = await _run_tool(tools, name, args, tool_calls_out, tool_results_out)
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": rendered})

    return {"text": text, "tool_calls": tool_calls_out, "tool_results": tool_results_out,
            "steps": steps}


async def _run_ollama_tool_loop(
    *,
    adapter: Any,
    config: dict[str, Any],
    model: str,
    instructions: str,
    tools: ToolRegistry,
    message: str,
    max_steps: int,
    model_settings: dict[str, Any],
) -> dict[str, Any]:
    """Ollama **原生**（``/api/chat``）的工具调用循环。

    与 OpenAI 兼容那条**只有三处**不同（其余逐行同构）：
    1. 响应读 ``message.content`` / ``message.tool_calls``（不是 ``choices[0].message``）；
    2. ``tool_calls[].function.arguments`` 是**对象**（OpenAI 是 JSON 字符串）⇒ ``_parse_tool_args``；
    3. 回灌用 ``{"role":"tool","tool_name":…,"content":…}``（**没有** ``tool_call_id``），
       用量字段是 ``prompt_eval_count``/``eval_count``（不是 ``usage``）。
    """
    timeout_ms, max_retries = _transport_retry(config)
    # ⚠️ 预算阈值必须跟着**实际**窗口走（num_ctx − 输出预留），见 _ollama_context_budget
    budget = _ollama_context_budget(config, model_settings)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": message},
    ]
    tool_calls_out: list[dict[str, Any]] = []
    tool_results_out: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    text = ""
    nudged = False

    for _ in range(max(1, int(max_steps))):
        # ⚠️ 上下文预算同 OpenAI 兼容那条（每步重发整份 messages，多步工具链会一路撑大）
        # strict：超窗就当场失败，不交给 Ollama 去静默截断（它 HTTP 200 且丢掉 system 开头）
        messages = _apply_context_budget(messages, steps, budget, strict=True)
        request = adapter.build_request(config, {
            "model": model,
            "messages": messages,
            "temperature": model_settings.get("temperature"),
            "maxTokens": model_settings.get("maxOutputTokens"),
        })
        if tools.tools:
            # Ollama 的 ``tools`` 就是 OpenAI 的 function 形状（同一份声明，无需转换）
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
        # ⚠️ Ollama 有一部分失败是 **HTTP 200 + {"error": …}**（如显存不够装下这个 num_ctx）
        # 而不是 4xx/5xx。此前直接取 ``message`` ⇒ 拿到空文本当作"模型什么都没说"收工，
        # 库里一行没变、接口却回 200。这里必须显式抛错。
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, str) and error.strip():
                raise ValueError(f"Ollama 返回错误（HTTP {response.status_code}）：{error.strip()}")
        usage = _ollama_usage(data)
        if usage is not None:
            steps.append({"usage": usage})

        assistant = data.get("message") if isinstance(data.get("message"), dict) else {}
        text = assistant.get("content") or ""
        calls = assistant.get("tool_calls") or []
        if not calls:
            if _need_write_nudge(tool_calls_out, text, nudged, tools):
                nudged = True
                steps.append({"nudge": "missing-write"})
                log_task_warn("Agent", "missing-write-nudge", {
                    "calledTools": _called_tools(tool_calls_out), "textChars": len(text),
                })
                messages.append({"role": "user", "content": _NUDGE_TEXT})
                continue
            break

        # ⚠️ assistant 这轮的消息（含 tool_calls）必须原样回灌，否则下一轮模型看不到自己调过什么
        messages.append({"role": "assistant", "content": assistant.get("content") or "",
                         "tool_calls": calls})
        for call in calls:
            function = call.get("function") or {}
            name = function.get("name")
            args = _parse_tool_args(function.get("arguments"))
            rendered = await _run_tool(tools, name, args, tool_calls_out, tool_results_out)
            messages.append({"role": "tool", "tool_name": name, "content": rendered})

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
    注入点（缺省时按 ``discover_mcp_tools()`` 懒连接发现，见 ``app/mcp/client.py``；失败只降级）。
    """
    options = options or {}
    built = build_agent_config(conn, type, episode_id, drama_id)
    if built is None:
        raise ValueError(f"Invalid agent type: {type}")

    driver: Generate = generate or default_generate
    sleep_fn = sleep or asyncio.sleep
    trace = start_trace("Agent", f"run-{type}",
                        {"agentType": type, "episodeId": episode_id, "dramaId": drama_id})

    # 统一组装：base + skill + 协议契约 + 风格注入 —— 与 build_agent_config **同一个函数**。
    # 这里刻意不复用 built.instructions：形参 instructions 可能是评测传入的候选提示词，
    # 而 style Profile 必须按「真实运行上下文」现算，否则评测结论对不上线上表现。
    skill_instructions = load_agent_skills(
        type, built.db_config.skills if built.db_config is not None else None
    )
    full_instructions = assemble_full_instructions(
        conn, type, episode_id, drama_id, instructions, skill_instructions,
    )

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

    只传**纯 baseInstructions**：skill、协议契约与风格注入统一由
    ``run_agent_with_instructions`` → ``assemble_full_instructions`` 组装，
    避免在两个地方各拼一遍（历史上正是这里只传 base，导致风格 Profile 从未进模型）。
    """
    built = build_agent_config(conn, type, episode_id, drama_id)
    if built is None:
        raise ValueError(f"Invalid agent type: {type}")
    return await run_agent_with_instructions(
        conn, type, episode_id, drama_id, built.base_instructions, message, options,
        generate=generate, sleep=sleep,
    )
