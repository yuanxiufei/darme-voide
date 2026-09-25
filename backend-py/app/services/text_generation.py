"""轻量文本生成（移植自 ``backend/src/services/text-generation.ts``，556 行）。

提供「一次性文本任务」的统一入口：动作建议、拆分镜头、续写剧本、优化视频提示词、
智能拆分角色视觉信息、音色角色打标。**它们都走 Provider Adapter 的文本链路**，
不引入 Mastra Agent + 协议契约的复杂度（那是 S5 的事）。

本文件里两类东西要分清：

* **纯逻辑**（占大半）：提示词拼装、``clean_json_string``、**本地规则拆分器**
  （8 张词表 + 相邻子句合并算法）、防幻觉校验、打标过滤 —— 全部可离线单测；
* **I/O**：``generate_text`` 里的取配置 + HTTP 调用 + 多模型 fallback。

⚠️ ``generate_text`` 会为**本地**配置申请 GPU 显存租约（``gpu_manager.acquire('text', …)``，
参与模型启动/卸载调度）—— 与 TS 一致：**每个模型尝试各自申请、``finally`` 里释放**
（fallback 到下一个模型会重新申请，不跨模型持有）。
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from sqlalchemy.engine import Connection

from ..core.response import js_truthy
from .adapters.registry import get_text_adapter
from .ai_configs import is_local_config
from .ai_providers import get_text_config
from .gpu_manager import gpu_manager
from .task_logger import log_task_error, log_task_progress, log_task_start, log_task_success
from .vendor_errors import fetch_with_retry, format_vendor_http_error

__all__ = [
    "ACTION_SYSTEM_PROMPT",
    "CONTINUE_RAW_SYSTEM_PROMPT",
    "CONTINUE_SCRIPT_SYSTEM_PROMPT",
    "OPTIMIZE_PROMPT_SYSTEM_PROMPT",
    "OPTIMIZE_TIMELINE_SYSTEM_PROMPT",
    "SPLIT_SYSTEM_PROMPT",
    "SPLIT_VISUALS_EXAMPLES",
    "SPLIT_VISUALS_SYSTEM_PROMPT",
    "VOICE_ROLE_TAGS",
    "VOICE_TAG_SYSTEM_PROMPT",
    "clean_json_string",
    "clean_visual_fragment",
    "continue_script",
    "generate_action_suggestion",
    "generate_text",
    "infer_voice_role_tags",
    "optimize_video_prompt",
    "split_character_visuals",
    "split_shot_into_sub_shots",
    "split_visuals_by_rules",
]


async def generate_text(
    conn: Connection,
    user_prompt: str,
    options: dict[str, Any] | None = None,
) -> str:
    """轻量文本生成：走 text 配置对应厂商，**支持多模型自动 fallback**。

    ``options`` 支持 ``temperature`` / ``maxTokens`` / ``system``。
    """
    options = options or {}
    config = get_text_config(conn)
    # ⭐ 自研引擎分支（2026-09-25 ✓）：provider=engine ⇒ 走**进程内**自研 LLM ✓（不 HTTP 调 ollama ✗）
    if (config.get("provider") or "").lower() == "engine":
        return await _generate_with_engine(config, user_prompt, options)
    models = (
        list(config["models"])
        if config.get("models") and len(config["models"]) > 0
        else [m for m in [config.get("model")] if js_truthy(m)]
    )
    if len(models) == 0:
        raise ValueError("文本模型未配置 — 请在设置中添加文本服务")

    adapter = get_text_adapter(config.get("provider"))
    is_local = is_local_config(config.get("baseUrl") or "", config.get("provider") or "")

    messages: list[dict[str, str]] = []
    if options.get("system"):
        messages.append({"role": "system", "content": options["system"]})
    messages.append({"role": "user", "content": user_prompt})

    last_error: Exception | None = None

    for model in models:
        # 本地 GPU 文本模型（如 Ollama qwen3）：申请显存租约，参与模型启动/卸载调度
        lease = None
        if is_local:
            lease = await gpu_manager.acquire("text", config.get("provider"), model,
                                              config.get("baseUrl"))
        try:
            request = adapter.build_request(
                config,
                {
                    "model": model,
                    "messages": messages,
                    "temperature": options.get("temperature"),
                    "maxTokens": options.get("maxTokens"),
                },
            )

            response = await fetch_with_retry(
                request["url"],
                {
                    "method": request["method"],
                    "headers": request["headers"],
                    # ⚠️ 发给厂商的请求体：Node 是 `JSON.stringify(request.body)`（紧凑）
                    "body": json.dumps(request["body"], ensure_ascii=False, separators=(",", ":"))
                    if request.get("body") is not None
                    else None,
                },
                "text",
                timeout_ms=180_000 if is_local else 60_000,
                max_retries=1 if is_local else 3,
            )

            if not response.is_success:
                last_error = ValueError(
                    format_vendor_http_error(response.status_code, response.text, "text")
                )
                log_task_error("TextGen", "http-error", {"model": model, "error": str(last_error)})
                continue

            data = response.json()
            content = adapter.parse_response(data)
            if not content:
                last_error = ValueError("文本模型返回为空")
                log_task_error("TextGen", "empty-response", {"model": model})
                continue

            log_task_success("TextGen", "done", {"model": model, "chars": len(content)})
            return content.strip()
        except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch (err: any) 等价
            last_error = err
            log_task_error("TextGen", "model-error", {"model": model, "error": str(err)})
        finally:
            # ⚠️ 与 TS 的 `finally { if (lease) lease.release() }` 一致：**每轮模型尝试各自释放**
            #    （fallback 到下一个模型时会重新申请，不会跨模型持有）
            if lease is not None:
                lease.release()

    if last_error is not None:
        raise last_error
    raise ValueError("文本生成失败")


#: 自研文本后端缓存 ✓（key = ``(gguf_path, tokenizer_path)`` ✓ —— 大权重只 load 一次 ✗）
_engine_backends: dict[tuple[str, str | None], Any] = {}


def _get_engine_backend(gguf_path: str, tokenizer_path: str | None) -> Any:
    """自研文本后端（缓存 + 幂等 load ✓）。"""
    from .engine import gguf, gguf_to_llm, llm_backend

    key = (gguf_path, tokenizer_path)
    backend = _engine_backends.get(key)
    if backend is not None:
        return backend
    info = gguf.inspect(gguf_path)
    config = gguf_to_llm.infer_llm_config(info.metadata)
    backend = llm_backend.LlmBackend(config, gguf_path=gguf_path, tokenizer_path=tokenizer_path)
    backend.load()
    _engine_backends[key] = backend
    return backend


async def _generate_with_engine(config: dict[str, Any], user_prompt: str,
                                options: dict[str, Any]) -> str:
    """provider=engine ⇒ 走**自研文本后端**（进程内推理 ✓ 不 HTTP 调 ollama ✗）。"""
    settings = config.get("settings") or {}
    gguf_path = settings.get("ggufPath") or settings.get("gguf_path")
    tokenizer_path = settings.get("tokenizerPath") or settings.get("tokenizer_path")
    if not gguf_path:
        raise ValueError("自研文本后端缺少 ggufPath（在文本服务 settings 里配 ✓）")
    backend = _get_engine_backend(str(gguf_path), str(tokenizer_path) if tokenizer_path else None)
    return await asyncio.to_thread(
        backend.generate, user_prompt,
        system=options.get("system"),
        temperature=float(options.get("temperature")
                          if options.get("temperature") is not None else 1.0),
        max_new_tokens=int(options.get("maxTokens") or 256),
    )


# ====== 分镜动作建议 ======

#: 分镜动作建议的 system prompt（**逐字与 TS 一致**，守卫会比对）
ACTION_SYSTEM_PROMPT = (
    "你是一名专业的短视频分镜导演。根据给定的镜头信息，为这个镜头设计 1-2 句简洁的"
    "中文运镜与动作建议，直接可用于视频生成提示词。要求："
    "（1）聚焦画面内的主体动作与镜头运动，例如推/拉/摇/移/跟、慢动作、特写推进等；"
    "（2）语言精炼，不超过 30 字；"
    "（3）只输出建议文本本身，不要解释、不要序号、不要引号。"
)


def _build_action_suggestion_prompt(input_data: dict[str, Any]) -> str:
    parts: list[str] = []
    if input_data.get("title"):
        parts.append(f"镜头标题：{input_data['title']}")
    if input_data.get("shotType"):
        parts.append(f"景别：{input_data['shotType']}")
    if input_data.get("angle"):
        parts.append(f"拍摄角度：{input_data['angle']}")
    if input_data.get("atmosphere"):
        parts.append(f"氛围：{input_data['atmosphere']}")
    scene_desc = input_data.get("imagePrompt") or input_data.get("description")
    if scene_desc:
        parts.append(f"画面内容：{scene_desc}")
    if input_data.get("action"):
        parts.append(f"已有动作描述：{input_data['action']}")
    if input_data.get("movement"):
        parts.append(f"已有运镜：{input_data['movement']}")

    if len(parts) > 0:
        return "\n".join(parts) + "\n\n请为以上镜头生成运镜与动作建议。"
    return "请为这个镜头生成运镜与动作建议。"


async def generate_action_suggestion(conn: Connection, input_data: dict[str, Any]) -> str:
    """生成分镜动作/运镜建议。"""
    user_prompt = _build_action_suggestion_prompt(input_data)
    log_task_start("ActionSuggestion", "generate", {"title": input_data.get("title") or ""})
    result = await generate_text(
        conn, user_prompt, {"system": ACTION_SYSTEM_PROMPT, "temperature": 0.8, "maxTokens": 120}
    )
    log_task_progress("ActionSuggestion", "generated", {"length": len(result)})
    return result


# ====== 拆分镜头 ======

SPLIT_SYSTEM_PROMPT = (
    "你是一位专业的电影分镜师。你的任务是把一个粗略的镜头描述，拆分为多个细致、专业的子镜头。"
    "每个子镜头只负责一个视角或动作细节，时长约 2-4 秒。"
    "合理运用远景、全景、中景、近景、特写等不同景别，子镜头之间保持叙事连贯。"
    "只输出 JSON，不要包含任何解释或 markdown 代码块标记。"
)


def clean_json_string(raw: str) -> str:
    """从模型输出里剥出 JSON：去 ```json 围栏，再截取首个 ``{`` 到末个 ``}``。"""
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return text


def _build_split_shot_prompt(input_data: dict[str, Any]) -> str:
    style_desc = input_data.get("visualStyle") or "电影写实风格"
    lines: list[str] = []
    scene_info = input_data.get("sceneInfo")
    if scene_info:
        lines.append(f"场景地点：{scene_info.get('location')}")
        lines.append(f"场景时间：{scene_info.get('time')}")
        if scene_info.get("atmosphere"):
            lines.append(f"场景氛围：{scene_info['atmosphere']}")
    if input_data.get("title"):
        lines.append(f"镜头标题：{input_data['title']}")
    if input_data.get("shotType"):
        lines.append(f"原始景别：{input_data['shotType']}")
    character_names = input_data.get("characterNames")
    if character_names and len(character_names) > 0:
        lines.append(f"出场角色：{'、'.join(character_names)}")
    lines.append(f"视觉风格：{style_desc}")
    scene_desc = input_data.get("description") or input_data.get("action")
    if scene_desc:
        lines.append(f"原始动作/画面描述：{scene_desc}")
    if input_data.get("dialogue"):
        lines.append(f"对白：{input_data['dialogue']}（请将对白放入最合适的子镜头，通常是角色说话的中景或近景）")
    lines.append("")
    lines.append("请将以上镜头拆分为 2-5 个子镜头，输出 JSON：")
    lines.append('{"subShots":[{"shotSize":"全景","cameraMovement":"静止","actionSummary":"60-100字的动作与画面描述","visualFocus":"视觉焦点"}]}')
    return "\n".join(lines)


def parse_sub_shots(raw: str) -> list[dict[str, Any]]:
    """解析并校验子镜头数组（三段错误文案与原实现逐字一致）。"""
    try:
        parsed = json.loads(clean_json_string(raw))
    except (ValueError, TypeError) as err:
        raise ValueError("AI 返回的拆分结果不是有效 JSON") from err

    sub_shots = parsed.get("subShots") if isinstance(parsed, dict) else None
    if not sub_shots or not isinstance(sub_shots, list) or len(sub_shots) == 0:
        raise ValueError("AI 拆分结果为空")
    for item in sub_shots:
        if not isinstance(item, dict) or not item.get("shotSize") or not item.get("actionSummary"):
            raise ValueError("子镜头缺少必要字段（shotSize / actionSummary）")
    return sub_shots


async def split_shot_into_sub_shots(conn: Connection, input_data: dict[str, Any]) -> list[dict[str, Any]]:
    """将一个粗略镜头拆分为多个子镜头。"""
    user_prompt = _build_split_shot_prompt(input_data)
    log_task_start("ShotSplit", "generate", {"title": input_data.get("title") or ""})
    result = await generate_text(
        conn, user_prompt, {"system": SPLIT_SYSTEM_PROMPT, "temperature": 0.8, "maxTokens": 2000}
    )
    sub_shots = parse_sub_shots(result)
    log_task_success("ShotSplit", "generated", {"count": len(sub_shots)})
    return sub_shots


# ====== 续写剧本 ======

CONTINUE_RAW_SYSTEM_PROMPT = (
    "你是一位专业的短剧编剧。根据用户提供的已有剧本内容，自然地续写后续剧情。要求："
    "（1）延续已有的文风、人称、叙事节奏与角色语气，保持前后一致；"
    "（2）剧情推进合理、有新情节发展，不重复已写内容；"
    "（3）若已有内容以对白或台词收尾，续写应包含新的情节推进或对白；"
    "（4）只输出续写正文本身，不要解释、不要\"续写：\"之类的前缀、不要加引号包裹。"
)

CONTINUE_SCRIPT_SYSTEM_PROMPT = (
    "你是一位专业的短剧编剧。根据用户提供的已有格式化剧本（场次剧本）内容，自然地续写后续场次。要求："
    "（1）严格延续已有的场次格式（场号、地点、时间、角色、对白/动作的排版结构）；"
    "（2）延续已有角色的性格与说话语气，剧情推进合理；"
    "（3）只输出续写的场次正文，不要解释、不要加 markdown 代码块、不要加\"续写：\"前缀。"
)

#: 续写时只把末尾这么多字符喂给模型（控制 token 且保证前后衔接）
_CONTINUE_TAIL_CHARS = 2400


def _build_continue_script_prompt(text: str) -> str:
    trimmed = (text or "").strip()
    tail = trimmed[-_CONTINUE_TAIL_CHARS:] if len(trimmed) > _CONTINUE_TAIL_CHARS else trimmed
    return f"以下是已有内容：\n\n{tail}\n\n请直接继续往下写，输出续写内容。"


async def continue_script(conn: Connection, input_data: dict[str, Any]) -> str:
    """AI 续写剧本（原始内容或格式化剧本）。"""
    trimmed = (input_data.get("text") or "").strip()
    if not trimmed:
        raise ValueError("内容为空，无法续写")

    system = (
        CONTINUE_SCRIPT_SYSTEM_PROMPT
        if input_data.get("mode") == "script"
        else CONTINUE_RAW_SYSTEM_PROMPT
    )
    user_prompt = _build_continue_script_prompt(trimmed)

    log_task_start("ContinueScript", "generate", {"mode": input_data.get("mode"), "length": len(trimmed)})
    result = await generate_text(conn, user_prompt, {"system": system, "temperature": 0.85, "maxTokens": 1200})
    log_task_success("ContinueScript", "generated", {"length": len(result)})
    return result


# ====== 优化视频提示词 ======

OPTIMIZE_PROMPT_SYSTEM_PROMPT = (
    "你是一位专业的 AI 视频生成提示词优化师。用户会给你一段视频生成提示词（可能简短、粗糙、缺少细节），"
    "以及该镜头的辅助信息（场景、角色、景别、运镜、风格等）。你的任务是把它们融合、扩写成一段专业、"
    "可直接用于视频生成模型的高质量中文提示词。要求："
    "（1）保留用户原始意图，不新增用户未指定的剧情或动作；"
    "（2）补全并细化画面主体、动作、镜头运动、景别、光线、氛围、节奏等电影化细节；"
    "（3）融合提供的场景地点/时间/氛围、出场角色名、视觉风格；"
    "（4）语言精炼流畅，总长控制在 200 字以内；"
    "（5）只输出优化后的提示词正文本身，不要解释、不要加「优化后」之类前缀、不要引号包裹。"
)

#: 分镜视频提示词（含时间轴分段 DSL）专用优化指令：
#: 分镜 agent 生成的 video_prompt 用 ``<n>`` 分隔时间段（0-3秒/3-6秒…），并用
#: ``<location>``/``<role>``/``<voice>`` 标签承载地点/角色/配音，这些是程序解析依赖的结构，
#: 优化时**严禁抹平**，只能对每个时间段内部的描述文字做润色扩写。
OPTIMIZE_TIMELINE_SYSTEM_PROMPT = (
    "你是一位专业的 AI 视频生成提示词优化师。用户会给你一段「分镜视频提示词」，它由多个时间段（例如「0-3秒」「3-6秒」）组成，"
    "用 <n> 作为分隔符，并可能包含 <location>（地点）、<role>（角色）、<voice>（配音）等结构化标签；"
    "这些时间段与标签是程序解析依赖的结构，必须原样保留。"
    "你的任务是：只对每个时间段内部的画面描述文字做扩写与润色，补全画面主体、动作、镜头运动、景别、光线、氛围、节奏等电影化细节，并融合辅助信息中的场景/角色/风格。"
    "硬性要求："
    "（1）所有时间段标注（如 0-3秒、3-6秒、6-9秒）原样保留，不得新增、删除、合并或改动时间段数量与顺序；"
    "（2）所有 <n> 分隔符、<location>…</location>、<role>…</role>、<voice>…</voice> 标签原样保留；"
    "（3）只扩写各段内的描述文字，不改变各段叙事顺序与剧情；"
    "（4）语言精炼，每段控制在 90 字以内，不新增用户未指定的剧情或动作；"
    "（5）只输出优化后的提示词正文本身，不要解释、不要加前缀、不要引号包裹。"
)

#: 时间轴 DSL 检测：命中才走「保留结构」的专用指令
_TIMELINE_DSL_RE = re.compile(
    r"<n\s*/?>|</?(?:location|role|voice)>|\d+\s*-\s*\d+\s*秒", re.IGNORECASE
)


def has_timeline_dsl(text: str) -> bool:
    """用户当前提示词是否为「分镜视频提示词」（含时间轴分段 DSL）。"""
    return _TIMELINE_DSL_RE.search(text or "") is not None


def _build_optimize_prompt(input_data: dict[str, Any]) -> str:
    lines: list[str] = []
    scene_info = input_data.get("sceneInfo")
    if scene_info:
        lines.append(f"场景地点：{scene_info.get('location')}")
        if scene_info.get("time"):
            lines.append(f"场景时间：{scene_info['time']}")
        if scene_info.get("atmosphere"):
            lines.append(f"场景氛围：{scene_info['atmosphere']}")
    character_names = input_data.get("characterNames")
    if character_names and len(character_names) > 0:
        lines.append(f"出场角色：{'、'.join(character_names)}")
    if input_data.get("title"):
        lines.append(f"镜头标题：{input_data['title']}")
    if input_data.get("shotType"):
        lines.append(f"景别：{input_data['shotType']}")
    if input_data.get("movement"):
        lines.append(f"运镜：{input_data['movement']}")
    if input_data.get("atmosphere"):
        lines.append(f"氛围：{input_data['atmosphere']}")
    if input_data.get("visualStyle"):
        lines.append(f"视觉风格：{input_data['visualStyle']}")
    scene_desc = input_data.get("description") or input_data.get("action")
    if scene_desc:
        lines.append(f"画面/动作描述：{scene_desc}")
    lines.append("")
    current = (input_data.get("currentPrompt") or "").strip()
    if current:
        lines.append(f"用户当前提示词：\n{current}")
    else:
        lines.append("用户当前未填写提示词，请根据以上信息生成。")
    lines.append("")
    lines.append("请输出优化后的视频生成提示词。")
    return "\n".join(lines)


async def optimize_video_prompt(conn: Connection, input_data: dict[str, Any]) -> str:
    """优化视频生成提示词（对齐 gcc KeyframeEditor 的 AI 优化）。

    注意：这是用户**主动点击「AI 优化」**触发的增强，与项目约定
    「审核失败时不自动改写提示词」无关。
    """
    log_task_start("OptimizePrompt", "generate", {"title": input_data.get("title") or ""})
    current = (input_data.get("currentPrompt") or "").strip()
    timeline = has_timeline_dsl(current)
    result = await generate_text(
        conn,
        _build_optimize_prompt(input_data),
        {
            "system": OPTIMIZE_TIMELINE_SYSTEM_PROMPT if timeline else OPTIMIZE_PROMPT_SYSTEM_PROMPT,
            "temperature": 0.7,
            "maxTokens": 900 if timeline else 400,
        },
    )
    log_task_success("OptimizePrompt", "generated", {"length": len(result)})
    return result


# ====== 智能拆分角色视觉信息 ======

SPLIT_VISUALS_SYSTEM_PROMPT = (
    "你是一位专业的角色视觉设定解析器。用户会给你一段角色的外貌特征描述，其中混杂了神态气质、外貌体型、服装穿着、武器装备、首饰配饰、随身器物等内容。\n"
    "你的任务：只精确提取以下三个视觉字段，其余内容一律忽略：\n"
    "（1）clothing 服装：身上穿着的衣物鞋帽（含服装配件），如「朴素古意的青色长衫」「玄色劲装」「白色西装」；\n"
    "（2）weapons 武器装备：随身携带或使用的武器（含法器），如「三尺青锋」「龙首紫檀法杖」「长弓」；\n"
    "（3）accessories 首饰配饰：佩戴在身上的饰品与发饰，如「简单束带」「白玉发簪」「蓝宝石项链」「玉佩」。\n"
    "拆分规则：\n"
    "1. 先通读全文，逐句扫描，凡涉及上述三类的信息必须全部提取，不得遗漏；\n"
    "2. 提取时必须保留原文对物体的完整描述：名物词、数量词与修饰语（如「朴素古意的青色长衫」「三尺青锋，剑鞘古朴无华，却蕴含着慑人的锋芒」），只去掉「身着/身穿/头戴/手持/腰间别着/身旁横放」等动词引导词；\n"
    "3. 不把神态、气质、环境、场景、动作过程带进任何字段，如「沉静如枯木」「与周围格格不入」「快逾闪电」应忽略；\n"
    "4. 武器配件（剑鞘/刀鞘/箭袋等）不单独拆成独立条目，但属于武器细节的描述（如「剑鞘古朴无华」「却蕴含着慑人的锋芒」）要保留在 weapons 字段中；\n"
    "5. 发带/发簪/耳环/项链/玉佩等佩戴类饰品一律归入 accessories，不归入 clothing；腰带/靴子/帽子归入 clothing；\n"
    "6. 法器/法杖/飞剑等进攻性器物归入 weapons；折扇、酒葫芦、拂尘、罗盘、乐器、书卷等非武器器物既不属于服装也不属于配饰，一律忽略、不输出；\n"
    "7. 同一类出现多项时用中文逗号分隔；原文未提及某类则输出空字符串，绝不编造；\n"
    '8. 只输出 JSON：{"clothing":"","weapons":"","accessories":""}，禁止 markdown 代码块和任何解释。'
)

#: 拆分 few-shot 示例：贴近真实长文本（神态/环境与装备混排），
#: 让模型模仿「只留名物短语」的输出风格。
SPLIT_VISUALS_EXAMPLES = (
    "示例 1：\n"
    "输入：「少年身着朴素古意的青色长衫，长发以简单束带松松挽于脑后，腰间别着一柄三尺青锋长剑，手中把玩着一把白面折扇。」\n"
    '输出：{"clothing":"朴素古意的青色长衫","weapons":"三尺青锋长剑","accessories":"简单束带"}\n\n'
    "示例 2：\n"
    "输入：「老者鹤发童颜，双目精光湛湛，一副仙风道骨模样。身穿玄色道袍，腰系玉带，脚蹬云纹靴。右手拄着一根龙首紫檀法杖，杖身刻满符文，隐隐泛着灵光，左手还托着一个刻满铭文的黄铜罗盘。左腕戴着一串佛珠，偶尔转动。」\n"
    '输出：{"clothing":"玄色道袍，玉带，云纹靴","weapons":"龙首紫檀法杖","accessories":"佛珠"}\n\n'
    "示例 3：\n"
    "输入：「她踩着高跟鞋走进会场，一袭白色晚礼服裙摆曳地，颈间坠着蓝宝石项链，耳垂上悬着细钻耳环。手提包里藏着一把袖珍手枪，作为最后防身之物。」\n"
    '输出：{"clothing":"白色晚礼服，高跟鞋","weapons":"袖珍手枪","accessories":"蓝宝石项链，细钻耳环"}\n\n'
    "示例 4：\n"
    "输入：「少年身着一袭朴素古意的青色长衫，长发以简单束带松松挽于脑后，身旁横放一柄三尺青锋，剑鞘古朴无华，却蕴含着慑人的锋芒，腰间别着一只温润的羊脂白玉酒葫芦。」\n"
    '输出：{"clothing":"朴素古意的青色长衫","weapons":"三尺青锋，剑鞘古朴无华，却蕴含着慑人的锋芒","accessories":"简单束带"}\n'
)

# ── 本地规则兜底：AI 输出异常或漏字段时按关键词从原文提取 ──
# 关键词先长后短：优先命中完整名物词，避免「衣/剑/枪」等单字提前匹配
SPLIT_CLOTHING_KEYWORDS = [
    "长衫", "汉服", "唐装", "戏服", "劲装", "布衣", "古衣", "战甲", "燕尾服", "西装", "礼服",
    "外套", "大衣", "风衣", "马甲", "衬衫", "卫衣", "牛仔裤", "短裤", "披风", "斗篷", "长裙",
    "围巾", "领带", "腰带", "护腕", "皮鞋", "运动鞋", "靴", "帽", "裙", "袍", "衫", "衣", "裤",
    "甲", "铠", "盔",
]
SPLIT_WEAPON_KEYWORDS = [
    "狼牙棒", "飞镖", "暗器", "火铳", "狙击枪", "冲锋枪", "机枪", "手枪", "步枪", "长剑", "青锋",
    "佩剑", "短剑", "匕首", "法杖", "长弓", "弓箭", "弓弩", "剑匣", "剑", "刀", "棍", "枪", "矛",
    "戟", "锤", "斧", "盾", "鞭", "锏", "弩", "杖",
]
#: 武器屏蔽词：命中时说明该句在描述「出剑/挥剑」等动作过程而非武器本体
SPLIT_WEAPON_BANNED = ["出剑", "拔剑", "挥剑", "使剑", "用剑", "剑尖", "剑光"]
#: 神态/环境/动作过程杂质词：命中时说明该子句与物品本体无关，合并邻近句时剔除
SPLIT_NOISE_KEYWORDS = [
    "气质", "神态", "神情", "神色", "眼神", "目光", "面色", "面容", "模样", "场景", "环境",
    "格格不入", "气势", "沉静", "淡漠", "精光", "不敢直视", "出剑", "拔剑", "挥剑", "使剑",
    "用剑", "剑尖", "剑光",
]
SPLIT_ACCESSORY_KEYWORDS = [
    "束发带", "发带", "发簪", "玉簪", "头冠", "凤冠", "王冠", "璎珞", "香囊", "荷包", "项链",
    "吊坠", "挂坠", "戒指", "扳指", "手串", "手链", "手镯", "脚链", "耳环", "耳钉", "花钿",
    "纶巾", "玉佩", "红绳", "簪", "钗", "冠", "束带",
]
#: 随身器物（非武器、非穿戴，如折扇/酒葫芦/罗盘）：
#: 不再作为独立字段输出，仅用于阻止其污染服装/武器/首饰的合并片段
SPLIT_PROP_NOISE_KEYWORDS = [
    "羊脂白玉酒葫芦", "芭蕉扇", "油纸伞", "紫金葫芦", "酒葫芦", "乾坤袋", "储物袋", "褡裢",
    "折扇", "团扇", "蒲扇", "羽扇", "拂尘", "罗盘", "花灯", "宫灯", "灯笼", "玉箫", "洞箫",
    "长箫", "横笛", "竹笛", "牧笛", "玉笛", "古琴", "瑶琴", "七弦琴", "焦尾琴", "琵琶", "古筝",
    "三弦", "书卷", "画卷", "书简", "卷轴", "经卷", "经书", "铜镜", "古镜", "香炉", "丹炉",
    "砚台", "算盘", "玉如意", "铜铃", "木鱼", "药瓶", "瓷瓶", "玉瓶", "酒坛", "茶壶", "酒樽",
    "金樽", "夜光杯", "水囊", "钱袋", "钱囊", "锦囊", "包袱", "药箱", "药篓", "手帕", "绣帕",
    "信笺", "令牌", "玉玺", "印玺", "官印", "圣旨", "如意",
]
#: 名物清洗时可去掉的常见动词/数量词引导与冗余后缀（先长后短）
SPLIT_REDUNDANT_PREFIXES = [
    "腰间别着一柄", "腰间别着", "腰间悬着一柄", "腰间悬着", "腰间佩着", "手里拿着", "手里拄着",
    "手执一柄", "手执", "手持一柄", "手持", "手中握着", "手上戴", "背后背着", "肩上扛着",
    "身旁横放一柄", "身旁横放", "横放", "头上戴", "头戴", "脚蹬一双", "脚蹬", "身着", "穿着",
    "身穿", "身披", "披着", "腰系", "腰缠", "长发以", "手腕上系着", "悬着", "挎着", "别着",
    "佩着", "戴着", "佩戴", "腰挂", "腰佩", "颈间坠着", "颈间戴着", "耳垂上悬着", "一手拄着",
    "拄着", "一串", "一柄", "一把", "一根", "一支", "一件", "一袭", "一身", "背着", "拿着",
    "握着", "脑后",
]
SPLIT_REDUNDANT_SUFFIXES = [
    "松松挽于脑后", "挽于脑后", "松松地挽着", "松松挽着", "松散地挽着", "斜挎在腰", "别在腰间",
    "挂在腰间", "挎在腰间", "握在手中", "提在手中", "垂在身侧", "拄于地面", "立在地上",
    "悬在腰间", "佩在腰间", "裙摆曳地", "裙摆轻摇", "无风自动", "随风飘动",
]

#: 相邻子句合并的长度上限（含分隔符）
_MERGE_MAX_CHARS = 34


def clean_visual_fragment(raw: str) -> str:
    """清洗单条视觉片段：去掉动词/数量词引导与冗余后缀，保留名物短语。

    兼容逗号分隔的多词值（逐段清洗后重新拼接，**不破坏分隔**）。
    """
    parts: list[str] = []
    for piece in re.split(r"[，,]", raw or ""):
        fragment = re.sub(r"[。；：、\n]", "", piece).strip()
        if not fragment:
            continue
        # 循环去掉引导词（如「穿着一身」→ 依次去掉「穿着」「一身」）
        changed = True
        while changed:
            changed = False
            for prefix in SPLIT_REDUNDANT_PREFIXES:
                if fragment.startswith(prefix):
                    fragment = fragment[len(prefix):].strip()
                    changed = True
                    break
        for suffix in SPLIT_REDUNDANT_SUFFIXES:
            if fragment.endswith(suffix):
                fragment = fragment[: -len(suffix)].strip()
                break
        parts.append(fragment)
    return "，".join(parts)


def split_visuals_by_rules(appearance: str) -> dict[str, str]:
    """本地规则拆分服装 / 武器 / 首饰（AI 失败时的兜底，也用于防幻觉校验）。"""
    sentences = [
        s.strip()
        for s in re.split(r"[，。；、\n,;：:！!？?]", appearance)
        if s.strip()
    ]

    def pick(keywords: list[str], banned: list[str] | None = None, other_keywords: list[str] | None = None) -> str:
        banned = banned or []
        other_keywords = other_keywords or []
        items: list[str] = []
        for index, sentence in enumerate(sentences):
            hit = next((k for k in keywords if k in sentence), None)
            if not hit:
                continue
            if any(b in sentence for b in banned) and len(hit) <= 2:
                continue
            # 同类已提取过 → 跳过（如原文重复粘贴同一武器）
            if any(hit in existing for existing in items):
                continue
            # 合并前后相邻的描述性子句：遇神态/环境/动作杂质词或他类关键词即停，
            # 保留完整物品描述（如「三尺青锋，剑鞘古朴无华，却蕴含着慑人的锋芒」）
            frag_parts = [sentence]
            total = len(sentence)

            def can_merge(text: str) -> bool:
                return not any(k in text for k in SPLIT_NOISE_KEYWORDS) and not any(
                    k in text for k in other_keywords
                )

            for j in range(index - 1, -1, -1):
                adjacent = sentences[j]
                if not can_merge(adjacent):
                    break
                if total + len(adjacent) + 1 > _MERGE_MAX_CHARS:
                    break
                total += len(adjacent) + 1
                frag_parts.insert(0, adjacent)
            for j in range(index + 1, len(sentences)):
                adjacent = sentences[j]
                if not can_merge(adjacent):
                    break
                if total + len(adjacent) + 1 > _MERGE_MAX_CHARS:
                    break
                total += len(adjacent) + 1
                frag_parts.append(adjacent)

            cleaned = clean_visual_fragment("，".join(frag_parts))
            if cleaned and cleaned not in items:
                items.append(cleaned)
        return "，".join(items)

    all_other = [
        *SPLIT_CLOTHING_KEYWORDS,
        *SPLIT_WEAPON_KEYWORDS,
        *SPLIT_ACCESSORY_KEYWORDS,
        *SPLIT_PROP_NOISE_KEYWORDS,
    ]
    return {
        "clothing": pick(SPLIT_CLOTHING_KEYWORDS, [], all_other),
        "weapons": pick(
            SPLIT_WEAPON_KEYWORDS,
            SPLIT_WEAPON_BANNED,
            [*SPLIT_CLOTHING_KEYWORDS, *SPLIT_ACCESSORY_KEYWORDS, *SPLIT_PROP_NOISE_KEYWORDS],
        ),
        "accessories": pick(
            SPLIT_ACCESSORY_KEYWORDS,
            [],
            [*SPLIT_CLOTHING_KEYWORDS, *SPLIT_WEAPON_KEYWORDS, *SPLIT_PROP_NOISE_KEYWORDS],
        ),
    }


def appears_in_source(value: str, appearance: str) -> bool:
    """AI 值是否**每个名物片段都能在原文中找到**（防「三尺青芒」类幻觉）。"""
    segments = [s.strip() for s in re.split(r"[，,]", value) if s.strip()]
    return len(segments) > 0 and all(len(seg) >= 2 and seg in appearance for seg in segments)


def _build_split_visuals_prompt(appearance: str) -> str:
    return SPLIT_VISUALS_EXAMPLES + f"现在拆分以下角色描述：\n{appearance}\n\n请严格按规则只输出 JSON。"


async def split_character_visuals(conn: Connection, input_data: dict[str, Any]) -> dict[str, str]:
    """从「外貌特征」文本智能拆分服装/武器/首饰三个字段。

    融合策略（保证完整描述 + 防幻觉）：

    1. **本地规则先拆**：名物短语精确、无动词引导/神态杂质（对常见词覆盖稳定）；
    2. **AI 增强**：AI 能保留原文完整物品描述（含剑鞘/锋芒等细节），覆盖规则词表盲区；
    3. **每字段「AI 优先，规则兜底」**：AI 值逐段校验**均来自原文**才采用，
       否则回退规则值。
    """
    appearance = (input_data.get("appearance") or "").strip()
    if not appearance:
        raise ValueError("外貌特征为空")

    rule_result = split_visuals_by_rules(appearance)

    log_task_start("SplitVisuals", "generate", {})
    ai: dict[str, Any] = {}
    try:
        result = await generate_text(
            conn,
            _build_split_visuals_prompt(appearance),
            {"system": SPLIT_VISUALS_SYSTEM_PROMPT, "temperature": 0.1, "maxTokens": 300},
        )
        parsed = json.loads(clean_json_string(result))
        ai = parsed if isinstance(parsed, dict) else {}
        log_task_progress("SplitVisuals", "ai", dict(ai))
    except Exception as err:  # noqa: BLE001
        # AI 输出异常 → 直接使用规则结果，保证按钮始终有输出
        log_task_progress("SplitVisuals", "fallback", {"reason": "ai-failed", "error": str(err)})

    # AI 值统一清洗：去掉动词引导/冗余后缀
    clean_ai = {
        "clothing": clean_visual_fragment(str(ai.get("clothing") or "")),
        "weapons": clean_visual_fragment(str(ai.get("weapons") or "")),
        "accessories": clean_visual_fragment(str(ai.get("accessories") or "")),
    }

    out = {
        key: (clean_ai[key] if appears_in_source(clean_ai[key], appearance) else rule_result[key])
        for key in ("clothing", "weapons", "accessories")
    }
    log_task_success("SplitVisuals", "generated", out)
    return out


# ====== 音色角色打标 ======

#: 音色角色类型标签（4 类，与前端 ``ROLE_TAGS`` 对齐）
VOICE_ROLE_TAGS = ("旁白", "主角", "反派", "配角")

VOICE_TAG_SYSTEM_PROMPT = (
    "你是专业的配音导演，擅长判断一个音色适合配音的角色类型。"
    "可选标签只有 4 个：旁白、主角、反派、配角。"
    "请根据音色的名称和官方描述判断，为每个音色打 1-3 个最贴切的标签。"
    '只输出 JSON，格式为 {"voice_id":["标签1","标签2"],...}，'
    "voice_id 必须与输入完全一致，不要包含任何解释或 markdown 代码块标记。"
)


def _build_voice_tag_prompt(voices: list[dict[str, Any]]) -> str:
    lines = []
    for voice in voices:
        desc = "、".join(voice.get("description") or [])
        line = f"- voice_id={voice.get('voiceId')}，名称={voice.get('voiceName')}"
        if desc:
            line += f"，描述={desc}"
        lines.append(line)
    joined = "\n".join(lines)
    return f"请为以下音色打角色类型标签：\n{joined}\n\n标签只能从「旁白 / 主角 / 反派 / 配角」中选择。"


def filter_voice_role_tags(parsed: dict[str, Any]) -> dict[str, list[str]]:
    """只保留合法标签且非空的条目（**空数组的键不输出**）。"""
    out: dict[str, list[str]] = {}
    for voice_id, tags in parsed.items():
        arr = [t for t in tags if str(t) in VOICE_ROLE_TAGS] if isinstance(tags, list) else []
        if arr:
            out[str(voice_id)] = arr
    return out


async def infer_voice_role_tags(conn: Connection, voices: list[dict[str, Any]]) -> dict[str, list[str]]:
    """批量推断音色适合的角色类型，返回 ``{voice_id: [标签]}``。

    用于音色库按角色类型筛选，替代前端纯正则推断。
    """
    if not voices:
        return {}

    log_task_start("VoiceTag", "generate", {"count": len(voices)})
    result = await generate_text(
        conn,
        _build_voice_tag_prompt(voices),
        {"system": VOICE_TAG_SYSTEM_PROMPT, "temperature": 0.1, "maxTokens": 2000},
    )

    try:
        parsed = json.loads(clean_json_string(result))
    except (ValueError, TypeError) as err:
        raise ValueError("AI 返回的打标结果不是有效 JSON") from err

    out = filter_voice_role_tags(parsed if isinstance(parsed, dict) else {})
    log_task_success("VoiceTag", "generated", {"tagged": len(out)})
    return out
