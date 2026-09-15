"""剧集时代背景 —— 移植 ``backend/src/services/era-background.ts``（136 行，**整域关闭**）。

概念：整部剧共享一个「时代背景」设定（朝代/世界观/环境/美术方向），由 AI 从剧本原文提炼后
存 ``dramas.era_background``（JSON 文本），在所有视觉资产生成前自动注入，
保证不同资产/镜头的时代感一致（不会出现古代戏里现代装饰这类串味）。

⚠️ 解析规则刻意与 TS 版**逐字对齐**（包括它比错误文案更宽松这一点）：
   实际判据是 ``summary || imageHint`` 非空，**不是**「era/summary/imageHint 三者齐全」，
   而路由层的 400 文案写的是后者。以代码为准，否则会拒掉 Node 侧本来接受的存量数据。

⚠️ 提炼（``extract_drama_era_background``）的三条保真点：

1. **剧本自动聚合无 ORDER BY**（原 TS 也没排序）⇒ 集数与顺序都按 SQLite 返回顺序拼
   ``第N集 标题\\n正文``，集与集之间空一行；
2. **8000 字截断是「首 70% + 中间省略 + 尾 30%」**（5600 + 2400），保留开头设定与结尾走向；
3. **异常一律包一层** ``时代背景提炼失败: ``（含 JSON 解析失败与字段缺失），
   而 ``Drama not found`` / ``该剧暂无剧本内容…`` 在 try **之外**，不被包装。
"""

from __future__ import annotations

import json
import re
from typing import Any

from sqlalchemy import and_, select, update
from sqlalchemy.engine import Connection

from ..core.models import dramas, episodes
from ..core.response import now
from .task_logger import log_task_error, log_task_progress, log_task_start, log_task_success

_EraBackground = dict[str, str]


def _js_truthy(value: Any) -> bool:
    """JS 的 falsy 集合：undefined / null / '' / 0 / NaN / false。"""
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, (int, float)):
        return value != 0
    return True


def _js_or(*values: Any) -> Any:
    """模拟 JS 的 ``a || b || c``：返回第一个 truthy 值，全 falsy 时返回最后一个。"""
    for v in values:
        if _js_truthy(v):
            return v
    return values[-1] if values else ""


def _js_str(value: Any) -> str:
    """模拟 JS 的 ``String(x)``。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        # TS 侧对对象会得到 "[object Object]"，但那属于脏输入；这里保留 JSON 便于排查。
        # 用紧凑分隔符，保持全仓 JSON 文本风格一致（Node 的 JSON.stringify 无空格）
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def parse_era_background(raw: str | None) -> _EraBackground | None:
    """解析 ``dramas.era_background``（JSON 文本）为结构化对象；空/非法返回 None。"""
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None

    out = {
        "era": _js_str(_js_or(obj.get("era"), obj.get("summary"), "")).strip()[:80],
        "summary": _js_str(_js_or(obj.get("summary"), obj.get("raw"), obj.get("era"), "")).strip()[:2000],
        # image_style_en 是历史键，优先级高于新键 imageHint（保持向后兼容）
        "imageHint": _js_str(
            _js_or(obj.get("image_style_en"), obj.get("imageHint"), obj.get("summary"), "")
        ).strip()[:1500],
    }
    if not out["summary"] and not out["imageHint"]:
        return None
    return out


def era_background_to_json(norm: _EraBackground) -> str:
    """落库用：与 TS 的 ``JSON.stringify(norm)`` 等价（键序固定为 era/summary/imageHint）。"""
    return json.dumps(norm, ensure_ascii=False, separators=(",", ":"))


def get_era_background(conn: Connection, drama_id: Any = None) -> _EraBackground | None:
    """读某剧的时代背景（``dramas.era_background`` 解析后）。

    ``if (!dramaId) return null`` 是 **JS 假值判断**：0 / 空串 / None 都返回 None。
    """
    if not drama_id:
        return None
    row = conn.execute(
        select(dramas.c.era_background).where(dramas.c.id == drama_id)
    ).first()
    if row is None:
        return None
    return parse_era_background(row[0])


def apply_era_image_clause(conn: Connection, prompt: str, drama_id: Any = None) -> str:
    """时代背景注入：``imageHint`` 存在时作为「时代/环境画面指令」**追加到 prompt 尾部**。

    不改变调用方原有 prompt 结构；无背景时原样返回。
    末尾点的处理：已有句号就不再加（``hint.endsWith('.') ? hint : `${hint}.```
    ``），最终拼成 ``"<prompt>, <hint>."``。
    """
    if not drama_id:
        return prompt
    era = get_era_background(conn, drama_id)
    hint = ((era or {}).get("imageHint") or "").strip()
    if not hint:
        return prompt
    clause = hint if hint.endswith(".") else f"{hint}."
    return f"{prompt}, {clause}"


#: 提炼时代背景的系统提示词（**逐字**对齐 TS 常量）
_ERA_EXTRACT_SYSTEM_PROMPT = (
    "你是资深影视美术指导。请从剧本中提炼整部剧的时代背景设定，输出会用于文生图模型保持全剧时代感一致。"
    "只输出 JSON，不要任何解释或 markdown 代码块标记，格式："
    '{"era":"时代标签（中文、简短，如 古代仙侠 / 现代都市·赛博朋克 / 民国谍战 / 中世纪奇幻）",'
    '"summary":"中文概述 60-120 字：世界观、地域、年代、社会风貌、常见场景等",'
    '"image_style_en":"英文 2-4 句画面指令：描述生成角色/服装/建筑/道具/环境图时必须体现的时代特征与'
    '美术风格方向，可包含材质/配色/年代细节词，不得描述任何具体人物"}'
    "若剧本包含架空/奇幻/科幻设定，优先交代其特殊规则（如灵力体系、机械义体）。"
)

#: 剧本输入截断上限（约 8000 字符，超出取首尾保设定与结局）
_MAX_SOURCE_CHARS = 8000

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*([\s\S]*?)```$")


def strip_json_fence(raw: str) -> str:
    """去掉 LLM 可能包裹的 ```json ``` 代码块，并截到最外层大括号内。"""
    text = (raw or "").strip()
    match = _FENCE_RE.match(text)
    if match:
        text = match.group(1).strip()
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        text = text[first:last + 1]
    return text


async def extract_drama_era_background(conn: Connection, drama_id: Any,
                                      source_text: str | None = None) -> _EraBackground:
    """从剧本原文 AI 提炼时代背景并落库 ``dramas.era_background``。

    ``source_text`` 未传时自动聚合该剧全部集数的 ``script_content``/``content``。
    """
    row = conn.execute(select(dramas).where(dramas.c.id == drama_id)).first()
    if row is None:
        # ⚠️ 这条在 try **之外** ⇒ 不会被包成「时代背景提炼失败: ...」
        raise ValueError("Drama not found")
    drama = row

    text = (source_text or "").strip()
    if not text:
        episode_rows = conn.execute(
            select(episodes).where(and_(episodes.c.drama_id == drama_id,
                                        episodes.c.deleted_at.is_(None)))
        ).all()
        # ⚠️ 原 TS **没有 ORDER BY**（顺序即 SQLite 返回顺序），这里照抄
        chunks: list[str] = []
        for episode in episode_rows:
            parts = [f"第{episode.episode_number}集"
                     + (f" {episode.title}" if episode.title else ""),
                     episode.script_content or episode.content or ""]
            chunks.append("\n".join([part for part in parts if part]))
        text = "\n\n".join(chunks).strip()

    if not text:
        raise ValueError("该剧暂无剧本内容，无法提炼时代背景。请先写剧本，或粘贴剧本原文后重试")

    log_task_start("EraBackground", "extract", {"dramaId": drama_id, "sourceChars": len(text)})

    truncated = text
    if len(truncated) > _MAX_SOURCE_CHARS:
        head = int(_MAX_SOURCE_CHARS * 0.7)
        tail = int(_MAX_SOURCE_CHARS * 0.3)
        truncated = f"{text[:head]}\n……（中间省略）……\n{text[-tail:]}"

    user_prompt = (f"剧名：{drama.title or '(未命名)'}\n以下是剧本内容：\n{truncated}\n\n"
                   f"请提炼这部剧的时代背景设定，只输出 JSON。")

    try:
        from .text_generation import generate_text  # noqa: PLC0415 —— 惰性导入避免环

        result = await generate_text(conn, user_prompt, {
            "system": _ERA_EXTRACT_SYSTEM_PROMPT,
            "temperature": 0.2,
            "maxTokens": 800,
        })
        log_task_progress("EraBackground", "ai-result",
                          {"dramaId": drama_id, "length": len(result)})
        obj = json.loads(strip_json_fence(result))
        parsed = parse_era_background(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
        if parsed is None:
            raise ValueError("AI 输出缺少有效时代背景字段")

        conn.execute(update(dramas)
                     .where(dramas.c.id == drama_id)
                     .values(era_background=era_background_to_json(parsed), updated_at=now()))
        log_task_success("EraBackground", "extract", {"dramaId": drama_id, "era": parsed["era"]})
        return parsed
    except Exception as exc:  # noqa: BLE001 —— 含 JSON 解析失败/字段缺失，一律包一层
        log_task_error("EraBackground", "extract", {"dramaId": drama_id, "error": str(exc)})
        raise ValueError(f"时代背景提炼失败: {exc}") from exc
