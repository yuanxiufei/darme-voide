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
from app.agent.evaluation.judge import judge_output
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


#: 工具不存在时运行时回灌的正文前缀（``runtime._run_tool`` 里那一句）。
_UNKNOWN_TOOL_PREFIX = "Unknown tool"


def _tool_error(rendered: Any) -> str | None:
    """从一条工具结果里取错误正文（没有错误 ⇒ ``None``）。

    工具输出是 ``normalize_tool_output`` 渲染的字符串，**未必是 JSON**（纯文本结果也合法）
    ⇒ 解析失败就当「没有结构化错误」，不臆造。
    """
    if isinstance(rendered, dict):
        payload = rendered
    elif isinstance(rendered, str):
        try:
            payload = json.loads(rendered)
        except ValueError:
            return None
    else:
        return None
    if isinstance(payload, dict) and payload.get("error"):
        return str(payload.get("error"))
    return None


def step_trace(
    tool_calls: list[dict[str, Any]] | None,
    tool_results: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """**单步对照**：把一整轮的轨迹拆成逐步的账（``(steps, stats)``）。

    为什么需要（2026-11-08，缺口 4 第三格的前置）：现有 case 是**整轮**打分，摘掉一个工具
    之后「第 3 步没工具可用」与「最后一句写得不好」扣的是同一个分 ⇒ 工具集换了也读不出
    是哪一处变了，leave-one-out 在这种粒度上判不准。

    这里只判**可机器判的三件事**（不猜语义）：

    * ``ok`` —— 该步有对应结果且结果里没有 ``error``（工具抛错、参数不合法都落在这一栏）；
    * ``unknownTool`` —— 该步撞上 ``Unknown tool``（**这就是「摘掉工具」的直接证据**：
      模型仍然想调它，而它不在本跑的暴露集里）；
    * ``ok=None`` —— 没有对应的工具结果（驱动没回灌）。**不判成失败**：没证据 ≠ 失败。

    ⚠️ 别把 ``ok`` 当「这一步有没有用」：工具成功返回 ``available:false``（语料/知识库
    目录缺失时的既定降级）也算 ok，那是「跑了但没料」，与「跑失败」是两件事。

    ⚠️ 输入**必须**是内存里那一跑的 `tool_calls`/`tool_results`（驱动回灌的数组），
    **不是**盘上的 `<dataRoot>/traces/*.jsonl`：2026-11-08 实测 61/61 条真实轨迹里
    `meta.toolCalls` 是**整数计数**，逐步明细根本没有落盘（`task_logger` 只记数量）。
    传进来一个计数也不炸（``stats["unavailable"]`` 如实说明"没有逐步明细"），
    但别指望从轨迹文件重建候选——候选只能从**基线那一跑的内存结果**取。
    """
    calls = tool_calls if isinstance(tool_calls, (list, tuple)) else []
    results = tool_results if isinstance(tool_results, (list, tuple)) else []
    unavailable = None
    if tool_calls is not None and not isinstance(tool_calls, (list, tuple)):
        unavailable = "toolCallsNotList"
        calls = []
    steps: list[dict[str, Any]] = []
    for index, raw in enumerate(calls):
        call = raw if isinstance(raw, dict) else {}
        name = str(call.get("toolName") or "")
        item = results[index] if index < len(results) else None
        if not isinstance(item, dict):
            item = None
        error = _tool_error(item.get("result")) if item is not None else None
        steps.append({
            "index": index,
            "tool": name,
            "ok": (error is None) if item is not None else None,
            "unknownTool": bool(error and error.startswith(_UNKNOWN_TOOL_PREFIX)),
            "error": error,
        })
    order: list[str] = []
    for step in steps:
        if step["tool"] and step["tool"] not in order:
            order.append(step["tool"])
    stats = {
        "total": len(steps),
        "ok": sum(1 for step in steps if step["ok"] is True),
        "failed": sum(1 for step in steps if step["ok"] is False),
        "unjudged": sum(1 for step in steps if step["ok"] is None),
        "unknownTool": sum(1 for step in steps if step["unknownTool"]),
        "tools": order,
        "unavailable": unavailable,
    }
    return steps, stats


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


def _report_steps(report: dict[str, Any], result: Any) -> dict[str, Any]:
    """把「单步对照」的账挂进报告（五个 ``kind`` 的**唯一**收口）。"""
    report["steps"], report["stepStats"] = step_trace(
        getattr(result, "tool_calls", None), getattr(result, "tool_results", None),
    )
    #: 模型的**最终正文**（2026-11-10 补）：语义判分的 ``[Output]`` 段只能取它 ——
    #: 确定性打分只看得见"产物里有什么"，看不见"模型怎么交代自己做了什么"。
    report["answer"] = str(getattr(result, "text", "") or "")
    return report


async def evaluate_case(
    conn: Connection,
    case_def: dict[str, Any],
    instructions: str,
    settings: dict[str, Any] | None = None,
    *,
    skills: str | None = None,
    tools: str | None = None,
) -> dict[str, Any] | None:
    """用指定 ``instructions`` 评测一个 case（确定性分 ＋ 可选语义分）。

    语义判分（P0-3）**只在 case 显式写了 ``judge`` 时**才跑：那五把规则尺子能判
    「有没有 12 个镜头」，判不了「这场戏的冲突够不够」—— 而提示词优化器的目标函数
    恰恰是后者。两者**并排**（``report["judge"]``），不互相替换：规则分回答
    "格式合不合规"，语义分回答"内容好不好"，混成一个数就再也拆不开。

    ⚠️ 开关是 ``judge`` 而**不是** ``criteria``：现存 case 的 ``criteria`` 是
    **规则词表**（``["保留人物关系", ...]``，给 ``scorer`` 做包含判定），拿它当判分标准
    会让每个 case 突然真跑一次 LLM 判分 —— 判分标准必须有人写（见 ``cases.py``）。
    """
    report = await _score_case(conn, case_def, instructions, settings,
                               skills=skills, tools=tools)
    if report is None:
        return None
    spec = _judge_spec(case_def)
    if spec is None:
        return report
    report["judge"] = (await judge_output(
        conn,
        criteria=spec["criteria"],
        output=str(report.get("answer") or ""),
        input_text=_judge_input(case_def),
        evidence=report.get("steps"),
        expected=spec.get("expected"),
        threshold=float(spec.get("threshold") or 0.6),
        # ⚠️ **不**透传被测那套 model_settings：判分的温度必须固定 0（``judge_output``
        # 的默认值），否则"同一份输出重复判分"会抖成两三个不同的分。
        options={},
    )).as_dict()
    return report


def _judge_spec(case_def: dict[str, Any]) -> dict[str, Any] | None:
    """``case_def["judge"]`` ⇒ 归一成 ``{"criteria", "expected", "threshold"}``。

    写成字符串是简写（它就是 ``criteria``）；写成 dict 时必须带非空 ``criteria``，
    否则**不判**（宁可明说"没标准"也不拿一个通用标准去评创作者的戏）。
    """
    raw = case_def.get("judge")
    if isinstance(raw, str):
        criteria = raw.strip()
        return {"criteria": criteria} if criteria else None
    if not isinstance(raw, dict):
        return None
    criteria = str(raw.get("criteria") or "").strip()
    if not criteria:
        return None
    return {
        "criteria": criteria,
        "expected": raw.get("expected"),
        "threshold": raw.get("threshold"),
    }


def _judge_input(case_def: dict[str, Any]) -> str:
    """``[Input]`` 段：case 给模型的任务输入（剧本/原始台词/角色表）。"""
    statement = case_def.get("statement") or {}
    parts: list[str] = []
    for key, label in (("script", "剧本"), ("content", "原始内容"),
                       ("prompt", "任务提示"), ("mode", "模式")):
        value = statement.get(key)
        if value:
            parts.append(f"{label}：{value}")
    for key in ("characters", "scenes"):
        value = statement.get(key)
        if value:
            parts.append(f"{key}：{value}")
    return "\n".join(parts)


async def _score_case(
    conn: Connection,
    case_def: dict[str, Any],
    instructions: str,
    settings: dict[str, Any] | None = None,
    *,
    skills: str | None = None,
    tools: str | None = None,
) -> dict[str, Any] | None:
    """确定性打分（五把规则尺子；``evaluate_case`` 的内核）。

    **不依赖、也不修改 DB 里的 ``agent_configs``**（候选提示词直接传入）。

    ``settings``：本次 run 的生成参数覆盖（``{"temperature": 0.0}``）。它只影响**这一跑**、
    不落库 —— 参数调优（``param_optimizer``）靠它做到「同一份提示词、只换温度」；
    不传就是 DB 里那套（老调用方行为零变化）。未知 ``kind`` 返回 ``None``（TS 的 switch
    没有 default ⇒ 落到 undefined）。

    ``skills``（2026-11-08 新增）：本次 run 的**技能绑定覆盖**（JSON 原文，形状同
    ``agent_configs.skills``）。与 ``settings`` 同一口径 —— 只影响这一跑、不落库，
    且**不写** ``skill_usage``（假设性绑定不许污染真实使用统计）。
    用途唯一：技能绑定优化（``skill_optimizer``）拿它做到「同一份提示词、只换注入的 skill 集」。
    不传（``None``）就是 DB 里那套；传 ``"[]"`` 是「一个都不注入」。

    ``tools``（2026-11-08 新增）：本次 run 的**工具集覆盖**（JSON 数组原文，元素是工具名）。
    与 ``settings`` / ``skills`` 同一口径 —— 只影响这一跑、不落库。用途唯一：
    工具集优化（``tool_optimizer``）做到「同一份提示词、只换暴露哪几个工具」。
    不传就是该 Agent 的全部工具；传 ``"[]"`` 是「一个工具都不给」（会退化成纯问答跑次）。

    **报告里的 ``steps`` / ``stepStats``** 是这一跑轨迹的逐步账（见 :func:`step_trace`），
    五个 ``kind`` 一律都带 —— 工具集那种「一步没用上」的差别只能在这个粒度上读出来。
    """
    # 空 dict 视同没传：``{"modelSettings": {}}`` 与不传在 runtime 里等价，但少一层误导
    run_options: dict[str, Any] = {"modelSettings": settings} if settings else {}
    # ⚠️ 用 ``is not None`` 判而不是真值判：``"[]"`` 是「全关」，它与「没传」是两件事
    if skills is not None:
        run_options["skillsRaw"] = skills
    if tools is not None:
        run_options["toolsRaw"] = tools

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
            return _report_steps(report, result)
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
            return _report_steps(report, result)
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
            return _report_steps(report, result)
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
            return _report_steps(report, result)
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
            return _report_steps(report, result)
        finally:
            cleanup_case(seed)

    return None
