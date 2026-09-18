"""**从现有表组装连续性计划**（适配器 ✓ 零依赖 ✓ 2026-09-18）。

## ⚠️ 本文件的一次**结论纠正**（值得留着看 ✓）

初版断言「契约 §6-§11 的字段**当前 schema 里没地方放**」✗ —— **这是错的** ✗✓：
我只去查了 ``storyboards`` / ``scenes`` 的**列** ✓，就下了结论 ✗，
**没有去找「有没有一张专用表」** ✗✗。而实际上 **``continuity_states`` 早就在那儿** ✓：

```text
continuity_states(episode_id, storyboard_id, scene_id,
                  state_type, entity_key, state_value, constraints, meta)
```

★ 这正是契约要的那种**通用状态表** ✓ —— 道具状态 ✓、画面方向 ✓、转场动机 ✓、
起止状态 ✓、线索可见 ✓ **全都能装** ✓。⇒ 需要加**表**吗？**不需要** ✓；
真正缺的是**词汇约定**（``state_type`` 该填什么值 ✓）—— 那是本模块现在给出的东西 ✓（见 ``STATE_TYPES`` ✓）。

> **教训**：**"schema 里没有"这种断言，必须查过"专用表"再说** ✗ ——
> 只查某几张表的列 ✓ 就宣布"无处安放" ✗ 是**典型的假结论** ✓（而且它会直接导致
> 去加一堆本不需要的表 ✗✗）。**先找有没有现成的家，再决定要不要盖房子** ✓。

## 本模块做两件事（都不撒谎 ✓）

1. **把 ``continuity_states`` 的通用行翻译成契约的 ``plan``** ✓ —— 按 ``state_type`` 分派 ✓
   （词汇表就是 :data:`STATE_TYPES` ✓，每条都标注契约出处 ✓）；
2. **报出每个契约要素的「数据覆盖率」** ✓✓ —— 例如「§8 只覆盖 3/12 镜」✓。
   这条很关键 ✗：**没有它，部分数据会得到"全绿"** ✓（看起来通过了 ✓ 其实只判了四分之一 ✓）。

## 三条铁律（照旧 ✓）

* **只映射真实存在的 ✓**：没有 ``continuity_states`` 行 ⇒ 那几项**就是没有** ✓，
  **绝不**补"状态没变"这类默认值 ✗（那会让 §8 **假绿** ✗✗，比不做还糟 ✓）；
* **判据只有一份** ✓：本模块**不判任何东西** ✗，判定仍由 :func:`app.services.continuity.check_continuity` 做 ✓；
* **判不了的就说判不了** ✓（覆盖率 ✓）。
"""
from __future__ import annotations

import json
from typing import Any, Iterable

__all__ = ["CONTRACT_MAP", "STATE_TYPES", "build_plan_from_rows", "coverage_report"]

#: ``continuity_states.state_type`` 的**词汇表** ✓（每条 = 一个契约要素 ✓）
#:
#: ⚠️ 这是本模块**新增的约定** ✓（原表没规定该填什么 ✓）——
#: 之所以敢定 ✓，是因为每个取值都能**逐条对上契约的小节** ✓，不是拍脑袋 ✓。
STATE_TYPES: dict[str, str] = {
    "prop": "§8 道具状态表 —— ``entity_key``=道具 id ✓，``state_value``=该镜状态（如 on_floor/hand）✓",
    "direction": "§6 场景空间表 —— ``entity_key``=角色 id ✓，``state_value``=left/right ✓",
    "transition": "§11 转场动机表 —— ``entity_key``=上一镜 storyboard_id ✓，"
                  "``state_value``=八类动机之一 ✓",
    "start_state": "§10/§11 接戏 —— ``entity_key``=实体 ✓，``state_value``=本镜开始时该实体的状态 ✓",
    "end_state": "§10/§11 接戏 —— 同上 ✓，本镜**结束**时的状态 ✓",
    "clue": "§9 线索/信息揭示表 —— ``entity_key``=线索 id ✓，``state_value``=visible/hidden ✓",
    "action": "§10 动作因果表 —— ``entity_key``=角色 id ✓，``state_value``=动作文本 ✓，"
              "``constraints``=effects 的 JSON 数组（五类作用 ✓）",
}

#: 契约要素 → 承载方式 ✓（``table=None`` 表示**仍然无处安放** ✗ —— 目前为空 ✓）
CONTRACT_MAP: tuple[dict[str, str], ...] = (
    {"contract": "§6 场景空间表（画面方向）", "carrier": "continuity_states / direction",
     "table": "continuity_states"},
    {"contract": "§8 道具状态表（时间线）", "carrier": "continuity_states / prop",
     "table": "continuity_states"},
    {"contract": "§9 线索/信息揭示表", "carrier": "continuity_states / clue",
     "table": "continuity_states"},
    {"contract": "§10 动作因果表", "carrier": "continuity_states / action",
     "table": "continuity_states"},
    {"contract": "§11 转场动机表", "carrier": "continuity_states / transition",
     "table": "continuity_states"},
    {"contract": "§10/§11 接戏（起止状态）", "carrier": "continuity_states / start_state·end_state",
     "table": "continuity_states"},
)

#: 仍然**无处安放**的契约要素 ✓（目前为空 ✓ —— 留机制 ✓ 不预置 ✗）
UNHOMED: tuple[dict[str, str], ...] = ()


def _as_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _field(row: Any, name: str) -> Any:
    if isinstance(row, dict):
        return row.get(name)
    return getattr(row, name, None)


def _json_list(value: Any) -> list[str]:
    """``constraints`` / ``meta`` 里的 JSON 数组 ✓（容错：不是 JSON 就当空 ✓）。"""
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    text = _as_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def _meta_dict(value: Any) -> dict[str, Any]:
    """``meta`` 里的 JSON 对象 ✓（容错：不是 JSON / 不是对象就当空 ✓）。"""
    if isinstance(value, dict):
        return value
    text = _as_text(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_plan_from_rows(storyboards: Iterable[Any], *,
                         scenes: Iterable[Any] = (),
                         characters: Iterable[Any] = (),
                         props: Iterable[Any] = (),
                         clues: Iterable[Any] = (),
                         continuity_states: Iterable[Any] = (),
                         storyboard_props: Iterable[Any] = ()) -> tuple[dict[str, Any], list[str]]:
    """把已有行翻译成连续性计划 ✓ ⇒ ``(plan, notes)``。

    ``continuity_states`` 是**关键**那张 ✓（通用状态 ✓，按 ``state_type`` 分派 ✓）；
    ``storyboard_props`` 只表示"这一镜有这件道具" ✓ —— 它**没有状态** ✗ ⇒
    它只用来统计覆盖率 ✓，**不会**被当成"状态未变" ✗✓。
    """
    notes: list[str] = []
    shot_rows = [row for row in storyboards]
    if not shot_rows:
        notes.append("这一集没有分镜行 ⇒ 没有可判定的内容 ✓（这不等于「通过」✗）")
        return {"shots": []}, notes

    # ── 基础表（只映射存在的列 ✓）────────────────────────────────────────
    plan: dict[str, Any] = {
        "locations": [{
            "id": _as_text(_field(row, "id")),
            "name": _as_text(_field(row, "location")) or _as_text(_field(row, "description")),
            "left": _as_text(_field(row, "left")),
            "right": _as_text(_field(row, "right")),
            "back": _as_text(_field(row, "back")),
            "foreground": _as_text(_field(row, "foreground")),
            "axis": _as_text(_field(row, "axis")),
            "time": _as_text(_field(row, "time")),
        } for row in scenes if _as_text(_field(row, "id"))],
        "characters": [{
            "id": _as_text(_field(row, "id")),
            "name": _as_text(_field(row, "name")),
            "appearance": _as_text(_field(row, "appearance")) or _as_text(_field(row, "core_features")),
            "current_state": _as_text(_field(row, "status")),
        } for row in characters if _as_text(_field(row, "id"))],
        "props": [{
            "id": _as_text(_field(row, "id")),
            "name": _as_text(_field(row, "name")),
            "first_shot": _as_text(_field(row, "first_shot")),
            "first_state": _as_text(_field(row, "status")),
        } for row in props if _as_text(_field(row, "id"))],
        "clues": [],
    }

    shots: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    #: 分镜原文里的动作**只是兜底** ✓（见下面「状态行是权威」✓）
    base_actions: dict[int, str] = {}
    for index, row in enumerate(shot_rows):
        raw_id = _as_text(_field(row, "id")) or _as_text(_field(row, "storyboard_number"))
        shot_id = f"sb{raw_id}" if raw_id else f"#{index + 1}"
        shot: dict[str, Any] = {"shot_id": shot_id}
        scene_ref = _as_text(_field(row, "scene_id"))
        if scene_ref:
            shot["location"] = f"L{scene_ref}"
        base_text = _as_text(_field(row, "action")) or _as_text(_field(row, "description"))
        if base_text:
            base_actions[len(shots)] = base_text
        shots.append(shot)
        if raw_id:
            by_id[f"sb{raw_id}"] = shot
    plan["shots"] = shots

    # ── ⭐ 通用状态行 → 契字段（按 state_type 分派 ✓）────────────────────
    unknown_types: set[str] = set()
    prop_shot_count = 0
    for row in continuity_states:
        shot_ref = _as_text(_field(row, "storyboard_id"))
        shot = by_id.get(f"sb{shot_ref}")
        kind = _as_text(_field(row, "state_type")).lower()
        key = _as_text(_field(row, "entity_key"))
        value = _as_text(_field(row, "state_value"))
        if shot is None:
            continue                       # 没挂到镜头的状态（例如集级）⇒ 已在覆盖率里体现 ✓
        if kind == "prop":
            shot.setdefault("prop_states", {})[key] = value
            prop_shot_count += 1
        elif kind == "direction":
            shot.setdefault("screen_direction", {})[key] = value
        elif kind == "transition":
            shot["transition"] = {"motive": value, "from": f"sb{key}" if key else ""}
        elif kind == "start_state":
            shot.setdefault("start_state", {})[key] = value
        elif kind == "end_state":
            shot.setdefault("end_state", {})[key] = value
        elif kind == "clue":
            if value.lower() in ("visible", "1", "true", "yes", "是"):
                shot.setdefault("clues_visible", []).append(key)
        elif kind == "action":
            entry: dict[str, Any] = {"actor": key, "action": value}
            effects = _json_list(_field(row, "constraints"))
            if effects:
                entry["effects"] = effects
            # ⚠️ **动作碰了哪件道具**必须能表达 ✓ —— 否则 §8 的「状态变化要有交代」
            #    永远判不了 ✓✗（自检 ⑲ 当场抓到：刀从 on_floor 变 hand ✓、
            #    同一镜也有动作 ✓，但**没人说那个动作碰的就是刀** ✗ ⇒ 判成"没交代"✓）。
            #    承载方式：``meta``（JSON ✓）里写 ``{"prop": "9"}`` 或 ``{"props": ["9","10"]}`` ✓。
            meta = _meta_dict(_field(row, "meta"))
            if meta.get("prop") not in (None, ""):
                entry["prop"] = _as_text(meta["prop"])
            props = _json_list(meta.get("props"))
            if props:
                entry["props"] = props
            shot.setdefault("actions", []).append(entry)
        else:
            unknown_types.add(kind)

    # ⚠️ **分镜原文的动作只是兜底** ✓：一旦这一镜有 ``state_type=action`` 行 ✓，
    #    就以**状态行为准** ✓ —— 否则同一个动作会被当成**两个**动作 ✓✗
    #    （自检 ⑳ 当场抓到：状态行带 effects ✓、原文那份没有 ✓ ⇒
    #     §10 会把这一镜判成「可删或可并」✗，而数据其实说了它推进剧情 ✓）。
    for position, text in base_actions.items():
        target = shots[position] if position < len(shots) else None
        if target is not None and not target.get("actions"):
            target["actions"] = [{"action": text}]     # ⚠️ 仍不编 effects ✓（见铁律 ✓）

    # 线索表：**首次可见镜头由可见行推出** ✓（最早那一镜的 `shot_id` ✓）——
    # 这样 §9 的「提前暴露」才判得动 ✓（否则 `first_visible_shot` 空 ⇒ 判不了 ✓）。
    first_seen: dict[str, str] = {}
    for shot in shots:
        for clue_id in shot.get("clues_visible") or []:
            first_seen.setdefault(str(clue_id), str(shot["shot_id"]))
    clue_rows = list(clues)
    plan["clues"] = [
        {"id": key,
         "name": next((_as_text(_field(row, "name")) for row in clue_rows
                       if _as_text(_field(row, "id")) == key), "") or key,
         "first_visible_shot": first_seen[key]}
        for key in first_seen
    ]

    if unknown_types:
        notes.append(f"⚠️ 有 {len(unknown_types)} 种 `state_type` 不在词汇表里 "
                     f"{sorted(unknown_types)} ✗ ⇒ 那几行**被忽略**了 ✓（请按 §STATE_TYPES 填 ✓）")

    notes.append(f"从 {len(shot_rows)} 个分镜行映射出 {len(shots)} 个镜头 ✓；"
                 f"`continuity_states` 分派完成 ✓")
    notes.append("⚠️ 缺的**一律不补默认值** ✗ —— 补「状态没变」会让 §8 假绿 ✓，比不做还糟 ✓")
    notes.extend(coverage_report(plan, storyboard_props=list(storyboard_props)))
    return plan, notes


def coverage_report(plan: dict[str, Any], *, storyboard_props: Iterable[Any] = ()) -> list[str]:
    """⭐ **每个契约要素的数据覆盖率** ✓ —— 没有它，部分数据会得到"全绿"✓✗。

    例：``§8 道具状态 3/12 镜`` ⇒ 那次判定**只覆盖 3 镜** ✓（用户据此知道该信多少 ✓）。
    """
    shots = plan.get("shots") or []
    total = len(shots) or 1

    def covered(predicate) -> int:
        return sum(1 for shot in shots if predicate(shot))

    linked = len({_as_text(_field(row, "storyboard_id")) for row in storyboard_props})
    rows: list[tuple[str, int, str]] = [
        ("§6 画面方向", covered(lambda s: s.get("screen_direction")), ""),
        ("§8 道具状态", covered(lambda s: s.get("prop_states")),
         (f"（另有 {linked} 镜登记了道具但**没有状态** ✗ ⇒ 那些镜的状态是**未知** ✓ "
          f"不是「没变」✗）" if linked else "")),
        ("§9 线索可见", covered(lambda s: s.get("clues_visible")), ""),
        ("§10 动作（带 effects）",
         covered(lambda s: any(a.get("effects") for a in s.get("actions") or [])), ""),
        ("§11 转场动机", covered(lambda s: s.get("transition")), "（第一镜不需要 ✓）"),
        ("接戏起止状态", covered(lambda s: s.get("start_state") or s.get("end_state")), ""),
    ]
    report = [f"覆盖率 {name} {count}/{total} 镜 ✓{suffix}" for name, count, suffix in rows]
    empty = [name for name, count, _suffix in rows if count == 0]
    if empty:
        report.append(f"⚠️ 这些要素**一镜都没有数据** ⇒ 对应判定**完全不成立** ✓：{empty} ✓"
                      f"（不是「通过」✗ —— 是「**没判**」✓）")
    return report
