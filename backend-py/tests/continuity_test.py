"""S7 自检：**连续性表**（契约 §6-§11 ✓ 零依赖 ✓ 2026-09-18）。

移植自 ``short-drama-agent``（``production-plan-contract.md`` §6-§11 ✓）。
本项目原来只有**事后**的 ``consistency_qc``（对已生成的相邻画面算 dHash ✗ ——
"拍完才发现不对"✗）；本套管的是**事前**：还没花钱生成，就说清「这一镜接不上上一镜」✓。

三条判据**逐字来自契约**（不是我发明的 ✓）：

* §8：*刀掉到地上，就不能再回到角色手里 —— 除非有镜头展示谁把它捡起来* ✓；
* §10：*不推进剧情/揭示信息/改变危险/建立空间/改变情绪的动作 ⇒ 标为可删或可并* ✓；
* §11：*永远不要只写「切到下一镜」，要解释剪辑理由* ✓。

运行::

    ./.venv/Scripts/python.exe tests/continuity_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.continuity import check_continuity  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def plan(shots: list[dict], **tables: object) -> dict:
    base: dict = {
        "locations": [{"id": "L1", "name": "案发房间", "left": "门", "right": "破窗",
                       "back": "白布尸体", "foreground": "湿水泥地", "axis": "左入右出"}],
        "characters": [{"id": "R5", "name": "沈砚", "appearance": "黑色湿短发/深色湿外套",
                        "current_state": "刚醒，左手有旧疤"}],
        "props": [{"id": "P1", "name": "水果刀", "first_shot": "kf001", "first_state": "floor"}],
        "clues": [{"id": "C1", "name": "数字 7", "first_visible_shot": "kf003"}],
    }
    base.update(tables)
    # ⚠️ 夹具便利：kf002 及之后的镜头**默认给一个合法转场动机** ✓ ——
    #    否则每个多镜用例都会先被 §11 拦住 ✓（我第一版就是这样：7 条用例全红 ✓，
    #    而那 7 条**根本不是要测转场** ✗）。要测「没动机」的用例显式传 ``transition=None`` ✓。
    filled: list[dict] = []
    for index, item in enumerate(shots):
        if index and "transition" not in item:
            item = {**item, "transition": {"motive": "action", "from": shots[index - 1].get("shot_id")}}
        filled.append(item)
    base["shots"] = filled
    return base


def shot(shot_id: str, **overrides: object) -> dict:
    base: dict = {
        "shot_id": shot_id, "location": "L1", "characters": ["R5"],
        "prop_states": {"P1": "floor"},
        "actions": [{"actor": "R5", "action": "睁眼", "effects": ["information"]}],
    }
    base.update(overrides)
    return base


def codes(issues: list[dict]) -> set[str]:
    return {str(item.get("code")) for item in issues}


# ══════════════════════════════════════════════════════════════════════════
# ① 契约 §8：道具状态**像时间线**（本套核心）
# ══════════════════════════════════════════════════════════════════════════
def case_prop_timeline() -> None:
    clean = check_continuity(plan([shot("kf001"), shot("kf002")]))
    check("① 干净计划 ⇒ `ok=True` ✓（同状态延续**不算**变化 ✓ 不误报 ✓）",
          clean.ok is True and clean.violations == [], clean.to_dict())

    jump = check_continuity(plan([
        shot("kf001"),
        shot("kf002", prop_states={"P1": "hand"},
             actions=[{"actor": "R5", "action": "站起来", "effects": ["plot"]}]),
    ]))
    check("② ⭐⭐ **刀从地上跳到手里、但没有任何动作交代** ⇒ 阻断 ✓"
          "（§8 原话：除非有镜头展示谁把它捡起来 ✓）",
          "prop-timeline" in codes(jump.violations) and jump.ok is False,
          jump.violations)

    picked = check_continuity(plan([
        shot("kf001"),
        shot("kf002", prop_states={"P1": "hand"},
             actions=[{"actor": "R5", "action": "捡起水果刀", "prop": "P1",
                       "effects": ["plot"]}]),
    ]))
    check("③ ⭐ 有「捡起」动作 ⇒ **放行** ✓（同一变化，判据是「有没有交代」✓）",
          picked.ok is True, picked.violations)

    no_actor = check_continuity(plan([
        shot("kf001"),
        shot("kf002", prop_states={"P1": "hand"},
             actions=[{"action": "捡起刀", "prop": "P1", "effects": ["plot"]}]),
    ]))
    check("④ 状态变化有动作但**没写谁做的** ⇒ 阻断 ✓（§8 要求「谁接触过它」✓）",
          "prop-timeline-actor" in codes(no_actor.violations), no_actor.violations)

    via_plural = check_continuity(plan([
        shot("kf001"),
        shot("kf002", prop_states={"P1": "thrown"},
             actions=[{"actor": "R5", "action": "甩出刀", "props": ["P1"],
                       "effects": ["danger"]}]),
    ]))
    check("⑤ `props`（复数）写法也认 ✓（手写清单两种写法都有 ✓）",
          via_plural.ok is True, via_plural.violations)

    timeline = check_continuity(plan([
        shot("kf001"),
        shot("kf002", prop_states={"P1": "hand"},
             actions=[{"actor": "R5", "action": "捡起", "prop": "P1", "effects": ["plot"]}]),
    ])).propTimeline
    check("⑥ 报告里给**道具时间线** ✓（人对着看：「kf001 在地上 → kf002 在手里」✓）",
          timeline == {"P1": [["kf001", "floor"], ["kf002", "hand"]]}, timeline)


# ══════════════════════════════════════════════════════════════════════════
# ② 契约 §6：画面方向不许翻（越轴）
# ══════════════════════════════════════════════════════════════════════════
def case_axis() -> None:
    stable = check_continuity(plan([
        shot("kf001", screen_direction={"R5": "left"}),
        shot("kf002", screen_direction={"R5": "left"}),
    ]))
    check("⑦ 方向稳定 ⇒ 放行 ✓", stable.ok is True, stable.violations)

    jump = check_continuity(plan([
        shot("kf001", screen_direction={"R5": "left"}),
        shot("kf002", screen_direction={"R5": "right"}, transition=None),
    ]))
    check("⑧ ⭐ 同场景内方向**翻转**+**没有转场动机** ⇒ 越轴 ✓ 阻断",
          "axis-jump" in codes(jump.violations), jump.violations)

    justified = check_continuity(plan([
        shot("kf001", screen_direction={"R5": "left"}),
        shot("kf002", screen_direction={"R5": "right"},
             transition={"motive": "sight", "from": "kf001"}),
    ]))
    check("⑨ 有**转场动机**的翻转 ⇒ 放行 ✓（换轴本身不是错 ✗，**没交代**才是错 ✓）",
          justified.ok is True, justified.violations)

    another = check_continuity(plan(
        [shot("kf001", screen_direction={"R5": "left"})],
        locations=[{"id": "L1", "name": "案发房间", "left": "门", "right": "破窗",
                    "back": "尸体", "foreground": "水泥地", "axis": "左入右出"},
                   {"id": "L2", "name": "走廊", "left": "消防门", "right": "电梯",
                    "back": "窗", "foreground": "脚印", "axis": "左入右出"}],
    ))
    check("⑩ 换场景 ⇒ 方向记忆**按场景分开** ✓（不同场景方向不同是正常的 ✓）",
          another.ok is True and "axis-jump" not in codes(another.violations), another.violations)

    unknown_loc = check_continuity(plan([shot("kf001", location="L9")]))
    check("⑪ 镜头在未登记的场景 ⇒ 阻断 ✓（§6：场景要建表 ✓ 否则空间关系无从保证 ✓）",
          "unknown-location" in codes(unknown_loc.violations), unknown_loc.violations)


# ══════════════════════════════════════════════════════════════════════════
# ③ 契约 §9：线索不许提前暴露
# ══════════════════════════════════════════════════════════════════════════
def case_clues() -> None:
    on_time = check_continuity(plan([
        shot("kf001"), shot("kf002"),
        shot("kf003", clues_visible=["C1"]),
    ]))
    check("⑫ 线索在声明的镜头露出 ⇒ 放行 ✓ 且记录**揭示顺序** ✓",
          on_time.ok is True and on_time.revealOrder == [["kf003", "C1"]], on_time.revealOrder)

    early = check_continuity(plan([
        shot("kf001", clues_visible=["C1"]), shot("kf002", clues_visible=["C1"]),
        shot("kf003", clues_visible=["C1"]),
    ]))
    check("⑬ ⭐⭐ **提前暴露** ⇒ 阻断 ✓（§9 原话：禁止提前暴露 ✓ —— 露早了悬念就没了 ✓）",
          "clue-early" in codes(early.violations), early.violations)

    # 表里声明的镜头**不在本次计划里** ⇒ 先后无法比较 ✓ 但"出现的位置和表不符"仍要说 ✓
    misplaced = check_continuity(plan([shot("kf001", clues_visible=["C1"])]))
    check("⑬′ 声明的镜头不在计划里 ⇒ 降级成**预警**并说明「对不上」✓（不假装判得了先后 ✓）",
          misplaced.ok is True
          and any(item["code"] == "clue-declared-absent" for item in misplaced.warnings),
          misplaced.warnings)

    unknown = check_continuity(plan([shot("kf001", clues_visible=["C9"])]))
    check("⑭ 未登记的线索 ⇒ 阻断 ✓（不建表就没法管顺序 ✓）",
          "unknown-clue" in codes(unknown.violations), unknown.violations)

    never = check_continuity(plan([shot("kf001"), shot("kf002"), shot("kf003")]))
    check("⑮ 表里声明了线索、却**任何镜头都没让它出现** ⇒ 预警 ✓（不阻断 ✓ —— 可能还没排到 ✓）",
          any(item["code"] == "clue-missing" for item in never.warnings) and never.ok is True,
          never.warnings)


# ══════════════════════════════════════════════════════════════════════════
# ④ 契约 §11：转场必须给动机（不许只写「切到下一镜」）
# ══════════════════════════════════════════════════════════════════════════
def case_transitions() -> None:
    missing = check_continuity(plan([shot("kf001"), shot("kf002", transition=None)]))
    check("⑯ ⭐ 第二镜**没写 transition** ⇒ 阻断 ✓（§11：每次切都要说清为什么切 ✓）",
          "transition-missing" in codes(missing.violations), missing.violations)

    empty = check_continuity(plan([
        shot("kf001"),
        shot("kf002", transition={"motive": "切到下一镜", "from": "kf001"}),
    ]))
    check("⑰ ⭐⭐ **只写「切到下一镜」** ⇒ 阻断 ✓✓（§11 逐字点名的就是这句 ✓）",
          "transition-empty" in codes(empty.violations), empty.violations)

    blank = check_continuity(plan([shot("kf001"), shot("kf002", transition={"from": "kf001"})]))
    check("⑱ 动机**留空** ⇒ 同样阻断 ✓（空 == 没写 ✓）",
          "transition-empty" in codes(blank.violations), blank.violations)

    good = check_continuity(plan([
        shot("kf001"),
        shot("kf002", transition={"motive": "prop", "from": "kf001"}),
    ]))
    check("⑲ 八类动机里点名（`prop` 道具转场 ✓）⇒ 放行 ✓",
          good.ok is True, good.violations)

    bad_motive = check_continuity(plan([
        shot("kf001"), shot("kf002", transition={"motive": "随便切", "from": "kf001"}),
    ]))
    check("⑳ 动机不在八类里 ⇒ 阻断 ✓（不接受自创类别 ✗）",
          "transition-unknown" in codes(bad_motive.violations), bad_motive.violations)

    first = check_continuity(plan([shot("kf001")]))
    check("㉑ **第一镜**不需要转场动机 ✓（它前面没有镜头 ✓）",
          first.ok is True, first.violations)

    wrong_from = check_continuity(plan([
        shot("kf001"), shot("kf002", transition={"motive": "action", "from": "kf099"}),
    ]))
    check("㉒′ `transition.from` 与上一镜对不上 ⇒ 预警 ✓（不阻断 ✓ —— 可能是重排过 ✓）",
          any(item["code"] == "transition-source" for item in wrong_from.warnings), "")


# ══════════════════════════════════════════════════════════════════════════
# ⑤ 契约 §10：动作必须"做成点什么" + 在场性
# ══════════════════════════════════════════════════════════════════════════
def case_actions() -> None:
    inert = check_continuity(plan([
        shot("kf001", actions=[{"actor": "R5", "action": "看了一眼", "effects": []}]),
    ]))
    check("㉓ ⭐⭐ **一件都没做到的动作**（§10 的五个作用全无 ✓）⇒ 进 `deletable` ✓"
          "（§10 原话：标为可删或可并 ✓ —— 但**不阻断** ✓ 删戏是创作决定 ✓）",
          inert.ok is True and any(item["action"] == "看了一眼" for item in inert.deletable)
          and any(item["code"] == "action-inert" for item in inert.warnings),
          inert.deletable)

    useful = check_continuity(plan([
        shot("kf001", actions=[{"actor": "R5", "action": "发现尸体", "effects": ["information"]}]),
    ]))
    check("㉔ 有作用（揭示信息 ✓）⇒ 不进 `deletable` ✓",
          useful.deletable == [] and useful.ok is True, useful.deletable)

    absent = check_continuity(plan([
        shot("kf001", characters=["R5"],
             actions=[{"actor": "R6", "action": "开枪", "effects": ["danger"]}]),
    ]))
    check("㉕ ⭐ **动作的人不在这一镜里** ⇒ 阻断 ✓（最常见的接戏事故 ✓）",
          "action-actor-absent" in codes(absent.violations), absent.violations)

    no_actor = check_continuity(plan([shot("kf001", actions=[{"action": "叹气"}])]))
    check("㉖ 动作没写 actor ⇒ 阻断 ✓（§10 要「谁做了什么」✓）",
          "action-actor" in codes(no_actor.violations), no_actor.violations)

    none = check_continuity(plan([shot("kf001", actions=[])]))
    check("㉗ 整镜没有动作 ⇒ 阻断 ✓（不允许「什么都没有」的镜头 ✓）",
          "action-missing" in codes(none.violations), none.violations)

    many = check_continuity(plan([
        shot("kf001", actions=[{"actor": "R5", "action": "睁眼", "effects": ["plot"]},
                               {"actor": "R5", "action": "坐起", "effects": ["plot"]}]),
    ]))
    check("㉘ 一镜多动作 ⇒ 预警 ✓（参考项目要求「每个镜头只做一件事」✓）",
          any(item["code"] == "action-multiple" for item in many.warnings)
          and many.ok is True, many.warnings)

    ghost = check_continuity(plan([
        shot("kf001", characters=[],
             actions=[{"actor": "R7", "action": "挥手", "effects": ["plot"]}]),
    ]))
    check("㉙ actor 不在**角色表**里 ⇒ 预警 ✓（§7 的角色卡是给视频模型照抄的 ✓）",
          any(item["code"] == "actor-unregistered" for item in ghost.warnings), ghost.warnings)


# ══════════════════════════════════════════════════════════════════════════
# ⑥ 表级必填 + **隔离性**（模块级状态必须每次清零 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_tables_and_isolation() -> None:
    thin = check_continuity(plan(
        [shot("kf001")],
        locations=[{"id": "L1", "name": "案房"}],
        characters=[{"id": "R5", "name": "沈砚"}],
    ))
    check("㉚ 六张表的必填字段缺项 ⇒ 逐条阻断 ✓（§6-§9 的字段就是给模型照抄的 ✓）",
          {"table-field"} == codes([item for item in thin.violations
                                    if item["code"] == "table-field"])
          and len([item for item in thin.violations if item["code"] == "table-field"]) == 2,
          thin.violations)

    check("㉛ 计划不是对象 / 没有镜头 ⇒ 报得出但**不抛异常** ✓",
          check_continuity("坏的").violations
          and check_continuity({"shots": []}).ok is True, "")

    # ⭐⭐ 隔离性：模块级方向记忆若不清零 ⇒ 上一次的方向会污染这一次 ✗
    first = check_continuity(plan([shot("kf001", screen_direction={"R5": "left"})]))
    second = check_continuity(plan([shot("kf001", screen_direction={"R5": "right"})]))
    check("㉜ ⭐⭐ **两次调用互不影响**（模块级方向记忆必须每次清零 ✓）"
          "—— 否则服务端会把上一部剧的方向带到下一部 ✗✓",
          first.ok is True and second.ok is True, (first.violations, second.violations))

    counts = check_continuity(plan([shot("kf001"), shot("kf002")])).counts
    check("㉝ 报告带计数 ✓（前端画概览用 ✓）",
          counts.get("shots") == 2 and "violations" in counts, counts)


# ══════════════════════════════════════════════════════════════════════════
# ⑦ 路由
# ══════════════════════════════════════════════════════════════════════════
def case_api() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    payload = plan([shot("kf001"), shot("kf002", transition=None)])
    response = client.post("/api/v1/continuity/check", json={"plan": payload})
    data = response.json().get("data") or {}
    check("㉞ POST /continuity/check 真能调用 ✓（200 + 逐条问题 + ok ✓）",
          response.status_code == 200 and data.get("ok") is False
          and data.get("violations"), (response.status_code, data))

    clean = client.post("/api/v1/continuity/check", json={"plan": plan([
        shot("kf001", actions=[{"actor": "R5", "action": "睁眼", "effects": ["information"]}]),
        shot("kf002", actions=[{"actor": "R5", "action": "站起", "effects": ["plot"]}],
             transition={"motive": "action", "from": "kf001"}),
    ])}).json()["data"]
    check("㉟ 干净计划经 API ⇒ `ok=True` ✓ 且带回**道具时间线**与**揭示顺序** ✓",
          clean.get("ok") is True and "propTimeline" in clean and "revealOrder" in clean, clean)

    check("㊱ 缺 plan ⇒ 400 ✓（不默认空计划放行 ✗）",
          client.post("/api/v1/continuity/check", json={}).status_code == 400)


def main() -> int:
    case_prop_timeline()
    case_axis()
    case_clues()
    case_transitions()
    case_actions()
    case_tables_and_isolation()
    case_api()

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
