"""**提示词生产工具**（占位符解析 + 五段质感层 ✓）。

移植自 ``reference/short-drama-agent``（``tagged-storyboard-format.md`` +
``mx-shell-workflow-adapter.md`` ✓）—— 这两件事本项目**原来没有** ✓：

* ``POST /api/v1/prompts/resolve`` —— 把 ``<location>L1</location>`` / ``<role>R5</role>`` /
  ``<duration-ms>6000</duration-ms>`` **解析成具体内容** ✓（本项目原有的
  ``strip_video_prompt_tags`` 只**剥标签**、会留下裸编号 ``L1`` ✗，而模型不认识内部编号 ✗）；
* ``POST /api/v1/prompts/polish`` —— 按 **Mx-Shell 五段式**逼出**物理锚点** ✓
  （真实镜头/调色/质感/瑕疵 ✓），并把「只有空话、没有物理细节」**逐条报出来** ✓。

⚠️ 两个端点都是**纯函数式**的 ✓（不碰 DB、不调模型、毫秒级 ✓）⇒ 前端可以在用户编辑提示词时
**实时**调它们 ✓（这也正是它们的用法 ✓）。
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from ..core.request_utils import read_json
from ..core.response import bad_request, success
from ..services import prompt_polish
from ..services import shot_placeholders as sp
from ..services.storyboard_helpers import segments_from_script

router = APIRouter(prefix="/api/v1/prompts", tags=["prompts"])


def _maps_from(body: Any) -> sp.PlaceholderMaps:
    """前端传的映射表 → :class:`PlaceholderMaps` ✓（缺项留空 ✓ 不报错 ✗）。"""
    if not isinstance(body, dict):
        return sp.PlaceholderMaps()
    def as_table(key: str) -> dict[str, str]:
        raw = body.get(key)
        if not isinstance(raw, dict):
            return {}
        return {str(k): str(v) for k, v in raw.items()}
    return sp.PlaceholderMaps(locations=as_table("locations"), roles=as_table("roles"),
                              props=as_table("props"), clues=as_table("clues"))


@router.post("/resolve")
async def resolve(request: Request) -> Any:
    """占位符 → **具体内容** ✓；``ok=False`` 表示**还不能直接送模型** ✗（原因在 ``problems`` ✓）。

    ``maps`` 里放四张表（``locations``/``roles``/``props``/``clues`` ✓）；
    返回 ``text`` ✓、``used`` / ``unresolved`` ✓、``durationsMs`` ✓、
    以及``residualTags`` / ``bareIds`` 两种"看不出来的坑" ✓。
    """
    body = await read_json(request)
    # ⚠️ 这里**不再**写 ``isinstance(body, dict)`` ✗ —— ``read_json`` 已经把非法 JSON / 非对象
    #    归一成 ``{}`` ✓（对齐 TS：由**业务校验**给出 400 ✓），那行是死代码 ✗。
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        return bad_request("缺少 text（要解析的提示词）")
    template = body.get("durationTemplate")
    return success(sp.resolve_placeholders(
        text, _maps_from(body.get("maps")),
        duration_template=str(template) if template else "目标时长{ms}毫秒",
        flag_bare_ids=bool(body.get("flagBareIds", True))).to_dict())


@router.post("/segments")
async def segments(request: Request) -> Any:
    """**导演稿 → 生成段** ✓（纯转换 ✓ 不碰 LLM / 库 / 盘 ✓；口径在 :mod:`app.services.engine.script_parse` ✓）。

    2026-09-24 接线 ✓：这套口径（5 种段标记 + 官方 ``[Shot N] At mm:ss`` + 无标记按 4.5 字/秒
    切 8~15 s + 时长往下吸附帧网格 ✓）此前**只有自检在调** ✗✗ ⇒ 前端与 agent 都拿不到 ✓。

    入参：``text``（导演稿 ✓）、``unitsPerSecond``（可选朗读语速 ✓ 默认 4.5 字/秒 ✓ ——
    ⚠️ 它**不是**「越界就拒」的参数 ✓：稿子多长就估多长 ✓，只影响估时 ✓）。

    出参：``segments``（逐段：文本 / 起止秒 / 帧数 / 断点类型 ✓）+ ``notes`` ✓ ——
    ⚠️ 显式 >15 s 的段会**按断点切开**（**不截断** ✗：截断会静默丢内容 ✓✗），切了几段在 ``notes`` 里说 ✓。
    """
    body = await read_json(request)
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        return bad_request("缺少 text（要分段的导演稿）")
    raw_units = body.get("unitsPerSecond")
    units: float | None = None
    if raw_units not in (None, ""):
        try:
            units = float(raw_units)
        except (TypeError, ValueError):
            return bad_request(f"unitsPerSecond 必须是数字（收到 {raw_units!r} ✗）")
        if not (units > 0):
            return bad_request(f"unitsPerSecond 必须大于 0（收到 {units} ✗）")
    return success(segments_from_script(text, units_per_second=units))


@router.post("/polish")
async def polish(request: Request) -> Any:
    """五段质感层 ✓ —— 输入是**连续性镜头卡**的字段 ✓（本层不新增剧情动作 ✗）。

    字段名按前端 camelCase 收 ✓（``themeTags``/``characterScene``/``lens``/``palette``/
    ``texture``/``shotType``/``movement``/``screenDirection``/``slice``/``sounds``/
    ``imperfections``/``handheld``/``innerMonologue``/``restrainedEnding`` ✓）。
    ``ok=False`` 时 ``issues`` 逐条说**哪一段约束不够硬** ✓（不是运行时错误 ✓）。
    """
    body = await read_json(request)
    # ⚠️ 这里**不再**写 ``isinstance(body, dict)`` ✗ —— ``read_json`` 已经把非法 JSON / 非对象
    #    归一成 ``{}`` ✓（对齐 TS：由**业务校验**给出 400 ✓），那行是死代码 ✗。

    def as_tuple(key: str) -> tuple[str, ...]:
        raw = body.get(key)
        if isinstance(raw, str):
            return (raw,) if raw.strip() else ()
        if isinstance(raw, list):
            return tuple(str(item) for item in raw if str(item).strip())
        return ()

    return success(prompt_polish.build_prompt_layer(prompt_polish.PolishInputs(
        theme_tags=as_tuple("themeTags"),
        character_scene=str(body.get("characterScene") or ""),
        lens=str(body.get("lens") or ""),
        palette=str(body.get("palette") or ""),
        texture=str(body.get("texture") or ""),
        shot_type=str(body.get("shotType") or ""),
        movement=str(body.get("movement") or ""),
        screen_direction=str(body.get("screenDirection") or ""),
        slice_text=str(body.get("slice") or body.get("sliceText") or ""),
        sounds=as_tuple("sounds"),
        imperfections=as_tuple("imperfections"),
        handheld=bool(body.get("handheld")),
        inner_monologue=bool(body.get("innerMonologue")),
        restrained_ending=bool(body.get("restrainedEnding")),
    )).to_dict())
