"""**连续性状态的写入侧**（`continuity_states` 的校验 + 幂等替换 ✓ 2026-09-18）。

## 这一步的**边界**（说清楚，免得以为它能自动补全 ✗）

`continuity_states` 要的东西里，**能从现有列自动推出的只有一小部分** ✗：

| 要素 | 能否从 DB 自动推出 | 说明 |
|---|---|---|
`action` | **部分** ✓ | 演员来自 `storyboard_characters` ✓、动作文本来自 `storyboards.action` ✓，但 **`effects`（五类作用）推不出来** ✗ |
`prop` 状态 | ✗ | `storyboard_props` 只说"这一镜有这件道具" ✓ **不含状态** ✗ |
`direction` / `transition` / `start·end_state` / `clue` | ✗ | 表里没有任何列能推 ✓ ⇒ **必须由生成阶段（LLM）给出** ✓ |

⇒ 所以本模块**只做机械侧** ✓：**校验 + 幂等写入** ✓（先把"能安全写进去"这件事建好 ✓）；
**LLM 侧（让分镜生成同时输出这些状态 ✓）是下一步** ✓ —— 本轮**不假装**它已经会自动生成 ✗。

## 四条语义（都是"写坏数据"最容易踩的 ✓）

1. ⭐ **词汇校验（拒收而非忽略）** ✓✓：`state_type` 必须在
   :data:`~app.services.storyboard_continuity.STATE_TYPES` 里 ✓，否则**报错并回显词汇表** ✓
   —— 静默忽略会让"明明填了却没生效"变成悬案 ✗；
2. ⭐ **幂等替换** ✓：同一镜重跑 ⇒ **先删该镜旧行再写** ✓（否则越积越多 ✗，
   而 :func:`~app.services.continuity.check_continuity` 会把旧行当"另一个状态"✗ ⇒ 假阳性 ✓）；
3. ⭐ **只动显式给出的镜** ✓：没提到的镜**一行不碰** ✓（免得手滑清掉整集 ✓）；
4. ⚠️ **归属校验** ✓：行里的 `storyboard_id` 必须**真属于**这一集 ✓
   （否则会把**别的集**的状态写坏 ✓✗ —— 这种错很难查 ✓）。
"""
from __future__ import annotations

from typing import Any, Iterable

from sqlalchemy import delete, insert, select
from sqlalchemy.engine import Connection

from ..core.models import continuity_states as states_tbl
from ..core.models import scenes as scenes_tbl
from ..core.models import storyboards as storyboards_tbl
from .storyboard_continuity import LEGACY_STATE_TYPES, STATE_TYPES

__all__ = ["STATE_TYPES_ALLOWED", "describe_vocabulary", "normalize_states",
           "write_episode_level_states", "write_episode_states", "write_scene_states",
           "write_shot_states"]

#: 词汇表（**大小写归一后**的形态 ✓ —— 与"归一化产物必须同源"那条规则一致 ✓）
#: ⚠️ **含既存的老名字** ✓：老数据/老写入方用的 `scene_space`/`prop_state`/`character_pose`
#: 必须**照收不拒** ✗（否则等于把既有数据判成非法 ✓✗）。新写入请用 :data:`STATE_TYPES` 的 7 条 ✓。
STATE_TYPES_ALLOWED: frozenset[str] = frozenset(
    kind.lower() for kind in (*STATE_TYPES, *LEGACY_STATE_TYPES))

#: 逐条必填 ✓（`constraints` / `meta` 可选 ✓ —— 它们承载 effects / prop 指认 ✓）
_REQUIRED: tuple[str, ...] = ("state_type", "entity_key", "state_value")


def describe_vocabulary() -> list[dict[str, str]]:
    """回显词汇表 ✓ —— 报错时一起给 ✓（"该怎么填"要能当场看到 ✓）。"""
    return [{"state_type": kind, "meaning": note} for kind, note in STATE_TYPES.items()]


def normalize_states(raw: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """校验 + 归一 ✓ ⇒ ``(rows, problems)``（**不抛异常** ✗ —— 全线报进 ``problems`` ✓）。"""
    problems: list[str] = []
    if not isinstance(raw, list):
        return [], ["states 必须是数组 ✗（每项一条状态 ✓）"]
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            problems.append(f"states[{index}] 不是对象 ✗")
            continue
        row = {key: ("" if item.get(key) is None else str(item.get(key)).strip())
               for key in (*_REQUIRED, "constraints", "meta")}
        missing = [key for key in _REQUIRED if not row[key]]
        if missing:
            problems.append(f"states[{index}] 缺少 {missing} ✗（必填：{list(_REQUIRED)} ✓）")
            continue
        kind = row["state_type"].lower()
        if kind not in STATE_TYPES_ALLOWED:
            problems.append(
                f"states[{index}] 的 state_type={row['state_type']!r} 不在词汇表里 ✗ ⇒ "
                f"**拒收**（静默忽略会让你以为填了却没生效 ✓）。可用："
                f"{sorted(STATE_TYPES_ALLOWED)} ✓")
            continue
        rows.append({**row, "state_type": kind})
    return rows, problems


def write_shot_states(conn: Connection, *, episode_id: int, storyboard_id: int,
                      states: Any) -> dict[str, Any]:
    """**单镜**幂等替换 ✓ —— 见模块文档的四条语义 ✓（不做事务提交 ✓ 由调用方 `get_tx` 决定 ✓）。"""
    rows, problems = normalize_states(states)
    storyboard_id = int(storyboard_id)
    episode_id = int(episode_id)

    # ⚠️ 归属校验：这一镜必须**真属于**这一集 ✓（写错集的状态很难查 ✓）
    owned = conn.execute(
        select(storyboards_tbl.c.id).where(storyboards_tbl.c.id == storyboard_id,
                                           storyboards_tbl.c.episode_id == episode_id)
    ).first()
    if owned is None:
        problems.append(f"storyboard {storyboard_id} 不属于 episode {episode_id} ✗ ⇒ "
                        f"**一条都不写**（否则会写坏别的集 ✓）")
        return {"ok": False, "problems": problems, "deleted": 0, "inserted": 0,
                "stateTypes": [], "vocabulary": describe_vocabulary()}

    # ⚠️⚠️ **有校验问题 ⇒ 本镜一行都不动** ✓✓（全有或全无 ✓）：
    #    初版是"先删旧行、再写通过校验的那几条" ✗ ⇒ 词汇填错时会把**这一镜原本的状态整片抹掉** ✗✗
    #    （自检 ⑩ 当场抓到 ✓）。**"拒收"必须是"什么都不做"** ✓ ——
    #    半套状态比没有更糟 ✗（体检会拿残缺的时间线当真 ✓）。
    if problems:
        problems.append("⇒ 本镜**一行都不动** ✓（全有或全无：半套状态比没有更糟 ✗）")
        return {"ok": False, "problems": problems, "deleted": 0, "inserted": 0,
                "stateTypes": [], "vocabulary": describe_vocabulary()}

    deleted = conn.execute(
        delete(states_tbl).where(states_tbl.c.episode_id == episode_id,
                                 states_tbl.c.storyboard_id == storyboard_id)
    ).rowcount or 0

    now_value = _now()
    if rows:
        conn.execute(insert(states_tbl), [
            {"episode_id": episode_id, "storyboard_id": storyboard_id,
             "scene_id": None, "state_type": row["state_type"],
             "entity_key": row["entity_key"], "state_value": row["state_value"],
             "constraints": row["constraints"], "meta": row["meta"],
             "created_at": now_value, "updated_at": now_value}
            for row in rows])

    return {"ok": not problems, "problems": problems, "deleted": int(deleted),
            "inserted": len(rows),
            "stateTypes": sorted({row["state_type"] for row in rows}),
            "vocabulary": describe_vocabulary() if problems else []}


def write_scene_states(conn: Connection, *, episode_id: int, scene_id: int,
                       states: Any) -> dict[str, Any]:
    """**按场景**幂等替换 ✓ —— 与 :func:`write_shot_states` 同语义 ✓。

    ⚠️ 为什么需要它（**我一开始漏了** ✗）：`continuity_states` 除了 `storyboard_id` 还有
    `scene_id` ✓，工具描述里也明写 *scene space layout* ✓ ⇒ **场景级状态是有意支持的** ✓；
    初版"只认镜头"把它一律拒掉 ✗ ⇒ 属**过度纠正** ✓（原有能力不该被顺手删掉 ✗）。
    仍**不做**的只有一件 ✓：**整集清空** ✗（那才是数据丢失的来源 ✓）。
    """
    rows, problems = normalize_states(states)
    scene_id = int(scene_id)
    episode_id = int(episode_id)

    owned = conn.execute(
        select(scenes_tbl.c.id).where(
            scenes_tbl.c.id == scene_id,
            # 场景的 `episode_id` 可空（剧级场景 ✓）⇒ 两种都算本集可用 ✓
            (scenes_tbl.c.episode_id == episode_id) | (scenes_tbl.c.episode_id.is_(None)))
    ).first()
    if owned is None:
        problems.append(f"scene {scene_id} 不属于 episode {episode_id} ✗ ⇒ **一条都不写**")
        return {"ok": False, "problems": problems, "deleted": 0, "inserted": 0,
                "stateTypes": [], "vocabulary": describe_vocabulary()}
    if problems:
        problems.append("⇒ 本场景**一行都不动** ✓（全有或全无 ✓）")
        return {"ok": False, "problems": problems, "deleted": 0, "inserted": 0,
                "stateTypes": [], "vocabulary": describe_vocabulary()}

    deleted = conn.execute(
        delete(states_tbl).where(states_tbl.c.episode_id == episode_id,
                                 states_tbl.c.scene_id == scene_id,
                                 states_tbl.c.storyboard_id.is_(None))
    ).rowcount or 0
    now_value = _now()
    if rows:
        conn.execute(insert(states_tbl), [
            {"episode_id": episode_id, "storyboard_id": None, "scene_id": scene_id,
             "state_type": row["state_type"], "entity_key": row["entity_key"],
             "state_value": row["state_value"], "constraints": row["constraints"],
             "meta": row["meta"], "created_at": now_value, "updated_at": now_value}
            for row in rows])
    return {"ok": True, "problems": [], "deleted": int(deleted), "inserted": len(rows),
            "stateTypes": sorted({row["state_type"] for row in rows}), "vocabulary": []}


def write_episode_level_states(conn: Connection, *, episode_id: int,
                               states: Any) -> dict[str, Any]:
    """**集级**状态（既没绑镜、也没绑场景 ✓）幂等替换 ✓。

    ⚠️ 为什么留这一档：老实现允许"集级状态" ✓（两个 id 都空 ✓），
    去掉它等于**顺手删掉既有能力** ✗ ⇒ 留着 ✓。
    但 ⚠️⚠️ 删除**只针对"两个 id 都为空"的那一桶** ✓✓ ——
    **不是**「按 episode_id 清空整集」✗（那才是原来的数据丢失来源 ✓）。
    """
    rows, problems = normalize_states(states)
    if problems:
        problems.append("⇒ 集级状态**一行都不动** ✓（全有或全无 ✓）")
        return {"ok": False, "problems": problems, "deleted": 0, "inserted": 0,
                "stateTypes": [], "vocabulary": describe_vocabulary()}

    deleted = conn.execute(delete(states_tbl).where(
        states_tbl.c.episode_id == int(episode_id),
        states_tbl.c.storyboard_id.is_(None),
        states_tbl.c.scene_id.is_(None))).rowcount or 0
    now_value = _now()
    if rows:
        conn.execute(insert(states_tbl), [
            {"episode_id": int(episode_id), "storyboard_id": None, "scene_id": None,
             "state_type": row["state_type"], "entity_key": row["entity_key"],
             "state_value": row["state_value"], "constraints": row["constraints"],
             "meta": row["meta"], "created_at": now_value, "updated_at": now_value}
            for row in rows])
    return {"ok": True, "problems": [], "deleted": int(deleted), "inserted": len(rows),
            "stateTypes": sorted({row["state_type"] for row in rows}), "vocabulary": []}


def write_episode_states(conn: Connection, *, episode_id: int, shots: Iterable[Any]) -> dict[str, Any]:
    """**批量**（每镜独立 ✓）：``shots = [{"storyboardId": 11, "states": [...]}, …]`` ✓。

    ⚠️ **只动显式给出的镜** ✓（没提到的镜一行不碰 ✓）；
    但**同一镜内**是"先删后写"✓ ⇒ 想追加而不是替换，请把旧行一起传进来 ✓。
    """
    results: list[dict[str, Any]] = []
    problems: list[str] = []
    if not isinstance(shots, (list, tuple)):
        return {"ok": False, "problems": ["shots 必须是数组 ✗"], "results": [],
                "deleted": 0, "inserted": 0}
    for index, shot in enumerate(shots):
        if not isinstance(shot, dict):
            problems.append(f"shots[{index}] 不是对象 ✗")
            continue
        raw_id = shot.get("storyboardId", shot.get("storyboard_id"))
        try:
            storyboard_id = int(raw_id)
        except (TypeError, ValueError):
            problems.append(f"shots[{index}] 的 storyboardId 不是整数：{raw_id!r} ✗")
            continue
        outcome = write_shot_states(conn, episode_id=episode_id, storyboard_id=storyboard_id,
                                    states=shot.get("states"))
        problems.extend(f"shots[{index}]: {item}" for item in outcome["problems"])
        results.append({"storyboardId": storyboard_id, "deleted": outcome["deleted"],
                        "inserted": outcome["inserted"], "stateTypes": outcome["stateTypes"]})
    return {
        "ok": not problems,
        "problems": problems,
        "results": results,
        "deleted": sum(item["deleted"] for item in results),
        "inserted": sum(item["inserted"] for item in results),
        "touchedShots": [item["storyboardId"] for item in results],
        "vocabulary": describe_vocabulary() if problems else [],
    }


def _now() -> str:
    """项目统一的时间戳格式 ✓（与 ``core.response.now()`` 保持一致 ✓ —— 不另造一种 ✓）。"""
    from ..core.response import now as _project_now
    return _project_now()
