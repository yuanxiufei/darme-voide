"""评测执行器 —— 与 ``evaluation/evaluator.ts``（210 行）对齐。

**seed 临时剧组 → 用指定提示词跑 Agent → 捕获工具调用 → 确定性评分 → 物理清理。**

* **隔离性**：每次评测都用**独立的临时 drama/episode**，跑完物理删除，不污染真实数据；
* **可重复**：同一 case + 同一提示词，评分应稳定（LLM 本身有随机性，属可接受方差）。

⚠️ 四处保真点：

1. **seed 必须逐条立即提交**（``engine.begin()`` 各开短事务）—— Node 的 drizzle
   ``db.insert(...).run()`` 是逐语句自动提交；若把 seed 攒在请求事务里，Agent 的工具
   开着**自己的**连接根本看不到这些行（SQLite 单写者，还会互相锁死）。同理 ``cleanup``。
2. **工具名要归一**（``norm_tool_name``：小写 + 去掉非字母数字）—— Mastra 回传的
   ``payload.toolName`` 可能是驼峰（``saveStoryboards``）而工具 id 是下划线
   （``save_storyboards``），不归一会**提取 0 命中**，分数直接归零且不报错。
3. **抽取口径各不相同**（别"统一"）：分镜与剧本取**最后一次**调用；角色/场景/音色是
   **跨多次调用累加**。
4. ``cleanup`` **只删本流程造出来的东西**，但**挂在分镜上的附表要删干净**：原实现照抄 TS
   只删 ``storyboard_characters``，把 ``storyboard_props`` / ``continuity_states`` 留成孤儿行
   （临时 drama 删了、孤儿行却永远留在库里；实测存量 39 条即由此累积）。现统一走
   ``delete_storyboard_children``。
"""
from __future__ import annotations

import json
from typing import Any

from sqlalchemy import delete as sql_delete, select
from sqlalchemy.engine import Connection

from app.core.db import engine
from app.core.models import (
    characters,
    dramas,
    episode_characters,
    episode_scenes,
    episodes,
    scenes,
    storyboards,
)
from app.core.response import now
from app.services.grid_payload import find_grid_payload
from app.services.storyboard_helpers import delete_storyboard_children
from app.agent.runtime import run_agent_with_instructions
from app.agent.evaluation.scorer import (
    score_extraction,
    score_grid_prompts,
    score_script_rewrite,
    score_storyboards,
    score_voice_assignment,
)

__all__ = [
    "EXTRACTOR_MESSAGE",
    "GRID_PROMPT_MESSAGE",
    "SCRIPT_REWRITER_MESSAGE",
    "STORYBOARD_MESSAGE",
    "VOICE_ASSIGNER_MESSAGE",
    "cleanup_case",
    "evaluate_case",
    "extract_characters",
    "extract_grid_payload",
    "extract_scenes",
    "extract_script",
    "extract_storyboards",
    "extract_voice_assignments",
    "grid_prompt_message",
    "norm_tool_name",
    "seed_case",
]

#: 四个 kind 各自喂给 Agent 的用户消息（**逐字与 TS 一致**）
STORYBOARD_MESSAGE = "请对当前集剧本进行分镜拆解并保存。"
EXTRACTOR_MESSAGE = "请提取当前集剧本中出现的角色和场景并保存。"
SCRIPT_REWRITER_MESSAGE = "请将当前集的原始内容改写为格式化剧本并保存。"
VOICE_ASSIGNER_MESSAGE = "请为当前剧组的角色分配合适的音色。"

#: 宫格这个 kind **没有**固定消息（TS 侧也没有）：镜头 id 与格数得现拼。
#: 措辞与 ``routers/grid.py::try_agent_grid_prompt`` 的 9 段拼装同源 —— 评测要走线上那条路，
#: 消息写另一套就是「评测的输入和生产的输入不是同一个东西」。参考图映射段此处恒空
#: （seed 的镜头没有出图 ⇒ 本来就没有参考图），按原口径**空段丢弃**。
GRID_PROMPT_MESSAGE = (
    "请为宫格图生成提示词，并优先调用工具完成。\n"
    "选中镜头ID：{shot_ids}\n"
    "行数：{rows}\n"
    "列数：{cols}\n"
    "模式：{mode}\n"
    "当提示词涉及到某个角色或场景时，直接把对应的图片编号写进提示词，"
    "例如：图片1中的角色A站了起来，图片3中的房间场景。不要只写名字，不写图片编号。\n"
    "必须严格按 {rows}x{cols} 生成，总共 exactly {total} visible panels。不要合并格子，不要缺格。\n"
    '必须返回 JSON，结构为：{{"grid_prompt":"...","cell_prompts":'
    '[{{"shot_number":1,"frame_type":"first_frame","prompt":"..."}}]}}'
)


def grid_prompt_message(
    shot_ids: list[int], rows: int, cols: int, mode: str,
) -> str:
    """按 ``GRID_PROMPT_MESSAGE`` 拼出这一跑的用户消息（镜头 id 走紧凑 JSON）。"""
    return GRID_PROMPT_MESSAGE.format(
        shot_ids=json.dumps(list(shot_ids), ensure_ascii=False, separators=(",", ":")),
        rows=rows,
        cols=cols,
        mode=mode,
        total=rows * cols,
    )


def norm_tool_name(name: Any) -> str:
    """工具名归一：小写 + 去掉所有非字母数字（驼峰与下划线因此等价）。"""
    return "".join(ch for ch in str(name or "").lower() if ch.isascii() and ch.isalnum())


def seed_case(
    *,
    case_id: str,
    script: str | None = None,
    content: str | None = None,
    characters_in: list[dict[str, Any]] | None = None,
    scenes_in: list[dict[str, Any]] | None = None,
    shots_in: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """建一套**临时**剧组数据，返回 ``{dramaId, episodeId, legal:{characterIds, sceneIds},
    shotIds:[…]}``。

    ⚠️ 每个 insert 一个短事务（见模块头第 1 点）。

    ``shots_in``（``grid_prompt`` 这个 kind 才用）把基准里的镜头落成**真实** ``storyboards``
    行 —— 宫格 Agent 的工具 ``read_shots_for_grid`` 只按 **id** 查库（它会再按
    ``storyboard_number`` 反查一次），把镜头直接塞进消息里它读不到。字段口径见
    ``types.STATEMENT_KEYS_BY_KIND["grid_prompt"]``。
    """
    ts = now()

    with engine.begin() as conn:
        drama_id = int(conn.execute(dramas.insert().values(
            title=f"[bench] {case_id}", status="draft", created_at=ts, updated_at=ts,
        )).lastrowid)

    with engine.begin() as conn:
        episode_values: dict[str, Any] = {
            "drama_id": drama_id, "title": "评测集",
            "content": content, "script_content": script,
            "created_at": ts, "updated_at": ts,
        }
        if "episode_number" in episodes.c:
            episode_values["episode_number"] = 1
        episode_id = int(conn.execute(episodes.insert().values(**episode_values)).lastrowid)

    character_ids: set[int] = set()
    for item in characters_in or []:
        with engine.begin() as conn:
            character_id = int(conn.execute(characters.insert().values(
                drama_id=drama_id, name=item.get("name"), role=item.get("role"),
                appearance=item.get("appearance"), personality=item.get("personality"),
                created_at=ts, updated_at=ts,
            )).lastrowid)
            character_ids.add(character_id)
            conn.execute(episode_characters.insert().values(
                episode_id=episode_id, character_id=character_id, created_at=ts,
            ))

    scene_ids: set[int] = set()
    for item in scenes_in or []:
        with engine.begin() as conn:
            # ⚠️ 场景同时挂 drama_id 与 episode_id（与常规流程不同，是 seed 的特意为之）
            scene_values: dict[str, Any] = {
                "drama_id": drama_id, "location": item.get("location"),
                "time": item.get("time"), "prompt": item.get("prompt"),
                "created_at": ts, "updated_at": ts,
            }
            if "episode_id" in scenes.c:
                scene_values["episode_id"] = episode_id
            scene_id = int(conn.execute(scenes.insert().values(**scene_values)).lastrowid)
            scene_ids.add(scene_id)
            conn.execute(episode_scenes.insert().values(
                episode_id=episode_id, scene_id=scene_id, created_at=ts,
            ))

    shot_ids: list[int] = []
    for item in shots_in or []:
        with engine.begin() as conn:
            shot_ids.append(int(conn.execute(storyboards.insert().values(
                episode_id=episode_id,
                # ``number`` 是基准里的镜头号；缺省按落库顺序补，避免 NN 约束炸掉整个 case
                storyboard_number=int(item.get("number") or (len(shot_ids) + 1)),
                title=item.get("title"),
                location=item.get("scene"),
                shot_type=item.get("shot_type"),
                action=item.get("action"),
                dialogue=item.get("dialogue"),
                created_at=ts, updated_at=ts,
            )).lastrowid))

    return {
        "dramaId": drama_id,
        "episodeId": episode_id,
        "legal": {"characterIds": character_ids, "sceneIds": scene_ids},
        "shotIds": shot_ids,
    }


def cleanup_case(seed: dict[str, Any]) -> None:
    """物理删除评测临时数据（seed 的 drama 及其全部关联，含 Agent 跑出来的分镜/角色/场景）。

    ⚠️ 失败**只告警不抛**（与 TS 的 catch 等价）—— 清理失败不该盖掉评测结果。
    """
    drama_id = seed["dramaId"]
    episode_id = seed["episodeId"]
    try:
        with engine.begin() as conn:
            storyboard_ids = [
                row[0] for row in conn.execute(
                    select(storyboards.c.id).where(storyboards.c.episode_id == episode_id)
                ).all()
            ]
            # 挂在分镜上的附表（角色/道具/镜头级状态）走单一真源，别只删角色关联 ——
            # 只删角色的版本会在库里攒下指向已不存在镜头的孤儿行（2026-10-30 实测存量 39 条）
            delete_storyboard_children(conn, storyboard_ids)
            conn.execute(sql_delete(storyboards).where(storyboards.c.episode_id == episode_id))
            conn.execute(sql_delete(episode_scenes).where(
                episode_scenes.c.episode_id == episode_id))
            conn.execute(sql_delete(episode_characters).where(
                episode_characters.c.episode_id == episode_id))
            conn.execute(sql_delete(scenes).where(scenes.c.drama_id == drama_id))
            conn.execute(sql_delete(characters).where(characters.c.drama_id == drama_id))
            conn.execute(sql_delete(episodes).where(episodes.c.id == episode_id))
            conn.execute(sql_delete(dramas).where(dramas.c.id == drama_id))
    except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch 等价
        print(f"[bench] cleanup failed: {err}")


def extract_storyboards(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """分镜产出：取**最后一次** ``save_storyboards`` 的 ``args.storyboards``。"""
    hits = [call for call in tool_calls
            if norm_tool_name(call.get("toolName")) == "savestoryboards"]
    last = hits[-1] if hits else None
    args = (last or {}).get("args") or {}
    return list(args.get("storyboards") or [])


def extract_characters(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """角色产出：**跨多次** ``save_dedup_characters`` 累加。"""
    collected: list[dict[str, Any]] = []
    for call in tool_calls:
        if norm_tool_name(call.get("toolName")) == "savededupcharacters":
            collected.extend((call.get("args") or {}).get("characters") or [])
    return collected


def extract_scenes(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """场景产出：**跨多次** ``save_dedup_scenes`` 累加。"""
    collected: list[dict[str, Any]] = []
    for call in tool_calls:
        if norm_tool_name(call.get("toolName")) == "savededupscenes":
            collected.extend((call.get("args") or {}).get("scenes") or [])
    return collected


def extract_script(tool_calls: list[dict[str, Any]]) -> str:
    """剧本产出：取**最后一次** ``save_script`` 的 ``args.content``。"""
    hits = [call for call in tool_calls if norm_tool_name(call.get("toolName")) == "savescript"]
    last = hits[-1] if hits else None
    return str(((last or {}).get("args") or {}).get("content") or "")


def extract_voice_assignments(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """音色产出：**跨多次** ``assign_voice`` 累加（只取三个字段）。"""
    collected: list[dict[str, Any]] = []
    for call in tool_calls:
        if norm_tool_name(call.get("toolName")) == "assignvoice":
            args = call.get("args") or {}
            collected.append({
                "character_id": args.get("character_id"),
                "voice_id": args.get("voice_id"),
                "reason": args.get("reason"),
            })
    return collected


def extract_grid_payload(result: Any) -> dict[str, Any] | None:
    """宫格产出：**先看工具结果、再看正文**（与 ``routers/grid.py`` 同序）。

    ⚠️ 解析走 ``services.grid_payload.find_grid_payload`` —— **单一真源**。在这里另写一遍
    "评测自己认的格式"，线上换一种写法时评测会照旧打高分，那是最典型的假绿。
    """
    return (find_grid_payload(getattr(result, "tool_results", None))
            or find_grid_payload(getattr(result, "text", None)))


async def evaluate_case(
    conn: Connection,
    case_def: dict[str, Any],
    instructions: str,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """用指定 ``instructions`` 评测一个 case，返回确定性评分报告（``caseId`` / ``runtimeModel`` 已填）。

    **不依赖、也不修改 DB 里的 ``agent_configs``**（候选提示词直接传入）。

    ``settings``：本次 run 的生成参数覆盖（``{"temperature": 0.0}``）。它只影响**这一跑**、
    不落库 —— 参数调优（``param_optimizer``）靠它做到「同一份提示词、只换温度」；
    不传就是 DB 里那套（老调用方行为零变化）。未知 ``kind`` 返回 ``None``（TS 的 switch
    没有 default ⇒ 落到 undefined）。
    """
    # 空 dict 视同没传：``{"modelSettings": {}}`` 与不传在 runtime 里等价，但少一层误导
    run_options: dict[str, Any] = {"modelSettings": settings} if settings else {}

    kind = case_def.get("kind")
    statement = case_def.get("statement") or {}
    rubric = case_def.get("rubric") or {}
    case_id = case_def.get("id")

    if kind == "storyboard":
        seed = seed_case(
            case_id=case_id, script=statement.get("script"),
            characters_in=statement.get("characters"), scenes_in=statement.get("scenes"),
        )
        try:
            result = await run_agent_with_instructions(
                conn, "storyboard_breaker", seed["episodeId"], seed["dramaId"],
                instructions, STORYBOARD_MESSAGE, run_options,
            )
            shots = extract_storyboards(result.tool_calls)
            report = score_storyboards(shots, rubric, seed["legal"])
            report["caseId"] = case_id
            report["runtimeModel"] = result.model
            return report
        finally:
            cleanup_case(seed)

    if kind == "extractor":
        seed = seed_case(case_id=case_id, script=statement.get("script"))
        try:
            result = await run_agent_with_instructions(
                conn, "extractor", seed["episodeId"], seed["dramaId"],
                instructions, EXTRACTOR_MESSAGE, run_options,
            )
            report = score_extraction(
                extract_characters(result.tool_calls), extract_scenes(result.tool_calls), rubric
            )
            report["caseId"] = case_id
            report["runtimeModel"] = result.model
            return report
        finally:
            cleanup_case(seed)

    if kind == "script_rewriter":
        seed = seed_case(case_id=case_id, content=statement.get("content"))
        try:
            result = await run_agent_with_instructions(
                conn, "script_rewriter", seed["episodeId"], seed["dramaId"],
                instructions, SCRIPT_REWRITER_MESSAGE, run_options,
            )
            report = score_script_rewrite(extract_script(result.tool_calls), rubric)
            report["caseId"] = case_id
            report["runtimeModel"] = result.model
            return report
        finally:
            cleanup_case(seed)

    if kind == "voice_assigner":
        seed = seed_case(
            case_id=case_id, script=statement.get("script"),
            # ⚠️ 音色 case 的角色**只带 name/role/personality**（不带 appearance）
            characters_in=[{
                "name": item.get("name"), "role": item.get("role"),
                "personality": item.get("personality"),
            } for item in (statement.get("characters") or [])],
        )
        try:
            result = await run_agent_with_instructions(
                conn, "voice_assigner", seed["episodeId"], seed["dramaId"],
                instructions, VOICE_ASSIGNER_MESSAGE, run_options,
            )
            report = score_voice_assignment(
                extract_voice_assignments(result.tool_calls), rubric, seed["legal"]
            )
            report["caseId"] = case_id
            report["runtimeModel"] = result.model
            return report
        finally:
            cleanup_case(seed)

    if kind == "grid_prompt":
        rows = int(statement.get("rows") or rubric.get("rows") or 0)
        cols = int(statement.get("cols") or rubric.get("cols") or 0)
        mode = str(statement.get("mode") or "first_frame")
        seed = seed_case(case_id=case_id, shots_in=statement.get("shots"))
        try:
            result = await run_agent_with_instructions(
                conn, "grid_prompt_generator", seed["episodeId"], seed["dramaId"],
                instructions, grid_prompt_message(seed["shotIds"], rows, cols, mode),
                run_options,
            )
            report = score_grid_prompts(extract_grid_payload(result), rubric)
            report["caseId"] = case_id
            report["runtimeModel"] = result.model
            return report
        finally:
            cleanup_case(seed)

    return None
