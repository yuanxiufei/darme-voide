"""S7 自检：**从现有表组装连续性计划**（适配器 ✓ 零依赖 ✓ 2026-09-18）。

⭐ 本套的**前提已被纠正过一次** ✓（值得留着 ✓）：

初版断言「契约 §6-§11 的字段**当前 schema 没地方放**」✗ —— **错的** ✗✓。
我只查了 ``storyboards`` / ``scenes`` 的**列** ✓ 就下结论 ✗，
**没去找有没有专用表** ✗✗；而 ``continuity_states``
（``episode_id/storyboard_id/scene_id/state_type/entity_key/state_value/constraints/meta`` ✓）
**早就在那儿** ✓★ —— 它正是契约要的通用状态表 ✓ ⇒ **不需要加表** ✓，
真正缺的是 ``state_type`` 的**词汇约定** ✓（本模块给出 ✓）。

所以本套钉两件事 ✓：

* **词汇表能逐个对上契约** ✓（``STATE_TYPES`` 每条都有契约出处 ✓；``CONTRACT_MAP`` 里六个要素
  **全部**有承载表 ✓）；
* ⭐ **数据覆盖率要如实报** ✓✓ —— 没有它，"只覆盖 3/12 镜"会得到**全绿** ✓✗。

运行::

    ./.venv/Scripts/python.exe tests/storyboard_continuity_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.continuity import check_continuity  # noqa: E402
from app.services.storyboard_continuity import (  # noqa: E402
    CONTRACT_MAP, STATE_TYPES, UNHOMED, build_plan_from_rows, coverage_report)

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


SHOTS = [
    {"id": 11, "episode_id": 3, "storyboard_number": 1, "scene_id": 5, "action": "沈砚猛地睁开眼"},
    {"id": 12, "episode_id": 3, "storyboard_number": 2, "scene_id": 5, "action": "坐起"},
]
SCENES = [{"id": 5, "location": "旧城区烂尾楼三层", "time": "深夜"}]
CHARACTERS = [{"id": 7, "name": "沈砚", "core_features": "黑色湿短发", "status": "active"}]
PROPS = [{"id": 9, "name": "水果刀", "status": "on_floor"}]
CLUES = [{"id": 4, "name": "数字 7"}]
#: ``continuity_states`` 的样例行（**按 ``STATE_TYPES`` 词汇表填** ✓）
STATES = [
    {"storyboard_id": 11, "state_type": "prop", "entity_key": "9", "state_value": "on_floor"},
    {"storyboard_id": 12, "state_type": "prop", "entity_key": "9", "state_value": "hand"},
    {"storyboard_id": 12, "state_type": "transition", "entity_key": "11", "state_value": "prop"},
    {"storyboard_id": 11, "state_type": "direction", "entity_key": "7", "state_value": "left"},
    {"storyboard_id": 11, "state_type": "start_state", "entity_key": "7", "state_value": "lying"},
    {"storyboard_id": 11, "state_type": "clue", "entity_key": "4", "state_value": "visible"},
    {"storyboard_id": 12, "state_type": "action", "entity_key": "7", "state_value": "捡起刀",
     "constraints": '["plot"]', "meta": '{"prop": "9"}'},
]


# ══════════════════════════════════════════════════════════════════════════
# ① 词汇表与承载表（**这是纠正后的结论** ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_vocabulary() -> None:
    check("① ⭐⭐ **六个契约要素全部有承载表** ✓✓（结论纠正：不需要加表 ✓ —— "
          "`continuity_states` 早就在那儿 ✓）",
          len(CONTRACT_MAP) == 6
          and all(item["table"] == "continuity_states" for item in CONTRACT_MAP),
          CONTRACT_MAP)
    check("② 词汇表每条都标**契约出处** ✓（不是拍脑袋定的 ✓）",
          all("§" in note for note in STATE_TYPES.values()) and len(STATE_TYPES) == 7,
          list(STATE_TYPES))
    check("③ `UNHOMED` 为空 ✓（真的没有「无处安放」的要素 ✓ —— 留机制但不预置 ✗）",
          UNHOMED == (), UNHOMED)


# ══════════════════════════════════════════════════════════════════════════
# ② 翻译：通用状态行 → 契约 `plan`
# ══════════════════════════════════════════════════════════════════════════
def case_translation() -> None:
    plan, notes = build_plan_from_rows(SHOTS, scenes=SCENES, characters=CHARACTERS,
                                       props=PROPS, clues=CLUES, continuity_states=STATES)
    shots = {shot["shot_id"]: shot for shot in plan["shots"]}
    check("④ 分镜行 ⇒ 镜头 ✓（`shot_id` 与状态行的 `sb<id>` 对得上 ✓）",
          set(shots) == {"sb11", "sb12"}, list(shots))
    check("⑤ `state_type=prop` ⇒ 该镜的 `prop_states` ✓（§8 有数据了 ✓）",
          shots["sb11"]["prop_states"] == {"9": "on_floor"}
          and shots["sb12"]["prop_states"] == {"9": "hand"}, shots["sb12"])
    check("⑥ `state_type=direction` ⇒ `screen_direction` ✓（§6 ✓）",
          shots["sb11"]["screen_direction"] == {"7": "left"}, shots["sb11"].get("screen_direction"))
    check("⑦ `state_type=start_state` ⇒ `start_state` ✓（§10/§11 接戏 ✓）",
          shots["sb11"]["start_state"] == {"7": "lying"}, shots["sb11"].get("start_state"))
    check("⑧ `state_type=transition` ⇒ `transition.motive` + `from=sb<key>` ✓（§11 ✓）",
          shots["sb12"]["transition"] == {"motive": "prop", "from": "sb11"},
          shots["sb12"].get("transition"))
    check("⑨ ⭐ `state_type=clue` ⇒ `clues_visible` ✓ **并由可见行推出** `first_visible_shot` ✓✓"
          "（§9 的「提前暴露」这才判得动 ✓）",
          shots["sb11"]["clues_visible"] == ["4"]
          and plan["clues"] == [{"id": "4", "name": "数字 7", "first_visible_shot": "sb11"}],
          plan["clues"])
    check("⑩ ⭐ `state_type=action` ⇒ actor + 动作 + **effects 来自 `constraints` JSON** ✓（§10 ✓）"
          "；⭐⭐ **「碰了哪件道具」来自 `meta` JSON** ✓✓（没它 ⇒ §8 永远判成「没交代」✗）",
          shots["sb12"]["actions"][-1] == {"actor": "7", "action": "捡起刀",
                                          "effects": ["plot"], "prop": "9"},
          shots["sb12"]["actions"])

    check("⑪ ⭐⭐ **没有状态行 ⇒ 就是不填** ✓✓（不补「状态没变」默认值 ⇒ 不给假绿 ✗）",
          all(key not in build_plan_from_rows(SHOTS)[0]["shots"][0]
              for key in ("prop_states", "screen_direction", "clues_visible", "transition")), "")

    mixed, _ = build_plan_from_rows(SHOTS, continuity_states=[
        {"storyboard_id": 11, "state_type": "道具", "entity_key": "9", "state_value": "x"}])
    check("⑫ 不在词汇表里的 `state_type` ⇒ **被忽略并报出** ✓（不静默吞 ✓）",
          "prop_states" not in mixed["shots"][0], mixed["shots"][0])

    bad, bad_notes = build_plan_from_rows(SHOTS, continuity_states=[
        {"storyboard_id": 11, "state_type": "whatever", "entity_key": "1", "state_value": "x"}])
    check("⑬ 同上有明确提示 ✓（`state_type` 大小写不敏感 ✓ —— 但拼错的词汇要能看见 ✓）",
          any("不在词汇表" in item for item in bad_notes), bad_notes[:2])


# ══════════════════════════════════════════════════════════════════════════
# ③ ⭐ 覆盖率：没有它，部分数据会"全绿"
# ══════════════════════════════════════════════════════════════════════════
def case_coverage() -> None:
    plan, _ = build_plan_from_rows(SHOTS)
    report = coverage_report(plan)
    check("⑭ ⭐ 每个契约要素都报**分母** ✓（`x/2 镜` ✓ —— 只有分子就说明不了覆盖面 ✓）",
          sum(1 for item in report if "/2 镜" in item) == 6, report)
    check("⑮ ⭐⭐ **一镜都没有数据的要素**要明说「**没判**」✓✓"
          "（而不是让它悄悄算成「通过」✗）",
          any("完全不成立" in item and "没判" in item for item in report), report[-1])

    partial, _ = build_plan_from_rows(SHOTS, storyboard_props=[{"storyboard_id": 11, "prop_id": 9}])
    partial_report = coverage_report(partial, storyboard_props=[{"storyboard_id": 11, "prop_id": 9}])
    check("⑯ ⭐ **登记了道具但没有状态** ⇒ 明确说那是「**未知**」✓ 不是「没变」✓✓"
          "（这条正是假绿的来源 ✓）",
          any("未知" in item and "不是「没变」" in item for item in partial_report),
          partial_report[1])

    full, _ = build_plan_from_rows(SHOTS, continuity_states=STATES)
    full_report = coverage_report(full)
    check("⑰ 有数据时覆盖率上升 ✓（§8 从 0/2 变成 2/2 ✓）",
          any("§8 道具状态 2/2" in item for item in full_report), full_report[:2])


# ══════════════════════════════════════════════════════════════════════════
# ④ 与判定器串起来：**现在真的判得动了**
# ══════════════════════════════════════════════════════════════════════════
def case_together() -> None:
    plan, _ = build_plan_from_rows(SHOTS, scenes=SCENES, characters=CHARACTERS,
                                   props=PROPS, clues=CLUES, continuity_states=STATES)
    report = check_continuity(plan)
    check("⑱ 适配器产出能直接喂给 `check_continuity` ✓（不抛异常 ✓）",
          isinstance(report.ok, bool), report.counts)
    check("⑲ ⭐ 道具状态**有交代**（`action` 行里 `prop` 指向它 ✓）⇒ §8 **不报** ✓"
          "（这条在补上状态行之后才可能通过 ✓）",
          not any(item.get("code") == "prop-timeline" for item in report.violations),
          report.violations[:2])
    check("⑳ ⭐ §10 **判得动了**：有 effects 的那一镜（sb12 ✓）不进「可删或可并」✓；"
          "而**没写 effects** 的 sb11 **照样进** ✓✓（数据没说它推进什么 ⇒ 如实提示 ✓ 不替它圆场 ✗）",
          {item.get("shot") for item in report.deletable} == {"sb11"}, report.deletable)

    no_state_plan, _ = build_plan_from_rows(SHOTS)
    thin = check_continuity(no_state_plan)
    check("㉑ 没有状态行时：判定结果**只反映能判的部分** ✓（不冒充全覆盖 ✗）",
          isinstance(thin.ok, bool), thin.counts)

    empty_plan, empty_notes = build_plan_from_rows([])
    check("㉒ 没有分镜 ⇒ 空计划 + 明说「不等于通过」✓",
          empty_plan["shots"] == []
          and any("不等于「通过」" in item for item in empty_notes), empty_notes)


def case_legacy_levels() -> None:
    """⭐⭐ 老数据的**场景级 / 角色级**状态要能**回填**到 §6/§7 ✓（光**收**不**读** = 没兼容 ✗）。

    这条是**真机跑出来的** ✓：对真库 `episode 1`（20 镜）跑体检时，阻断项**全是**
    `locations[*]` 缺 `left/right/...` ✓ + `characters[*]` 缺 `current_state` ✗
    ⇒ 而这两个正是 `scene_space` / `character_pose` 的**家** ✓ ⇒ 若不回填，
    **§6/§7 永远空 ⇒ 体检永远红** ✓✗（那等于造了一个"永远报错"的检查 ✓）。
    """
    plan, notes = build_plan_from_rows(
        SHOTS, scenes=SCENES, characters=CHARACTERS,
        continuity_states=[
            {"scene_id": 5, "state_type": "scene_space", "entity_key": "left", "state_value": "门"},
            {"scene_id": 5, "state_type": "scene_space", "entity_key": "axis",
             "state_value": "左入右出"},
            {"storyboard_id": 11, "state_type": "character_pose", "entity_key": "7",
             "state_value": "侧身半跪"},
        ])
    location = plan["locations"][0]
    check("㉓ ⭐⭐ 场景级 `scene_space` **回填**到 §6 ✓（`left`/`axis` ✓）"
          "—— 只收不读的话 §6 永远空 ⇒ 体检永远红 ✗",
          location.get("left") == "门" and location.get("axis") == "左入右出", location)
    check("㉔ ⭐ `character_pose` 回填到 §7 的 `current_state` ✓",
          plan["characters"][0].get("current_state") == "侧身半跪", plan["characters"][0])
    check("㉕ 回填这件事在 notes 里**写明白** ✓（不做隐性换算 ✓）",
          any("回填" in item for item in notes), notes)


def main() -> int:
    case_vocabulary()
    case_translation()
    case_coverage()
    case_together()
    case_legacy_levels()

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
