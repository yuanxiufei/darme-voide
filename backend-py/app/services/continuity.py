"""**连续性表**（移植 ``reference/short-drama-agent`` 的 ``production-plan-contract.md`` §6-§11 ✓ 零依赖 ✓）。

## 为什么需要它（本项目原来只有"事后看图" ✗）

本项目已有的 ``consistency_qc`` 是**事后**的：对**已生成的相邻画面**算 dHash 相似度 ✓
—— 那是"**拍完**才发现不对" ✗。缺的是**事前**的：把六张表摆出来，
在**还没花钱生成**的时候就说清「这一镜接不上上一镜」✓。

对应契约的六节 ✓：

| 节 | 表 | 本模块查什么 |
|---|---|---|
§6 | 场景空间表 | 同场景内**画面方向不许翻**（越轴 ✗）+ 必填空间关系齐备 ✓ |
§7 | 角色连续性表 | 角色卡必填（**这是给视频模型用的连续性卡** ✓ 不是人物小传 ✗） |
§8 | 道具状态表 | ⭐ **道具像时间线**：状态**变了就必须有镜头交代** ✓ |
§9 | 线索/信息揭示表 | ⭐ **不许提前暴露** ✓（揭示顺序单调 ✓） |
§10 | 动作因果表 | ⭐ **什么都不推动的动作 ⇒ 可删或可并** ✓ |
§11 | 镜头转场动机表 | ⭐ **永远不许只写「切到下一镜」** ✓（八类动机必须点名 ✓） |

## 三条**照抄契约**的判断（不自己发明规则 ✓）

* §8 原话：*如果刀掉到地上，它就不能再回到角色手里 —— 除非有一个镜头展示了那个人把它捡起来* ✓
  ⇒ 状态变化**必须**由同一镜里的一个动作（涉及该道具 ✓）来交代 ✓；
* §10 原话：*如果一个动作没有推进剧情、揭示信息、改变危险、建立空间或改变情绪，
  就把它标为可删或可并* ✓ ⇒ 五个作用一个都给不出的动作进 ``deletable`` ✓（预警 ✓ 不阻断 ✓）；
* §11 原话：*永远不要只写「切到下一镜」，要解释剪辑理由* ✓
  ⇒ 动机必须落在八类里 ✓，空动机或被判为「无动机」的一律报 ✓。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["ACTION_EFFECTS", "TRANSITION_MOTIVES", "TABLE_FIELDS", "ContinuityReport",
           "check_continuity"]

#: 四张**表级**必填字段 ✓（§6/§7/§8/§9 ✓ —— 缺了就没法做连续性判断 ✓）
TABLE_FIELDS: dict[str, tuple[str, ...]] = {
    "locations": ("id", "name", "left", "right", "back", "foreground", "axis"),
    "characters": ("id", "name", "appearance", "current_state"),
    "props": ("id", "name", "first_shot", "first_state"),
    "clues": ("id", "name", "first_visible_shot"),
}

#: 八类转场动机 ✓（§11 原文枚举 ✓ —— **不许**用「切到下一镜」这类空话 ✗）
TRANSITION_MOTIVES: dict[str, str] = {
    "sight": "视线转场", "action": "动作转场", "sound": "声音转场", "prop": "道具转场",
    "emotion": "情绪转场", "information": "信息转场", "space": "空间转场", "danger": "危险转场",
}

#: 一个动作至少要**做到**这五件事之一 ✓（§10 原文 ✓）；一件都做不到 ⇒ 可删/可并 ✓
ACTION_EFFECTS: dict[str, str] = {
    "plot": "推进剧情", "information": "揭示信息", "danger": "改变危险",
    "space": "建立空间", "emotion": "改变情绪",
}

#: 「没有动机」的常见写法 ✓ —— 写这些等于没写 ✓（§11 点名的就是它 ✓）
_EMPTY_MOTIVES: frozenset[str] = frozenset(
    {"", "切", "切镜", "切到下一镜", "下一镜", "继续", "延续", "接着", "cut", "next"})


@dataclass
class ContinuityReport:
    """连续性体检 ✓（``violations`` **阻断** ✓；``warnings`` 只是建议 ✓）。"""

    violations: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    #: 道具时间线 ✓：``{道具: [(镜头, 状态), ...]}``（人对着看的 ✓ 机器判据的依据 ✓）
    propTimeline: dict[str, list[list[str]]] = field(default_factory=dict)  # noqa: N815
    #: 线索揭示顺序 ✓：``[(镜头, 线索), ...]``
    revealOrder: list[list[str]] = field(default_factory=list)  # noqa: N815
    #: §10 的「可删或可并」✓（**不阻断** ✓ —— 删戏是创作决定，由人拍板 ✓）
    deletable: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """**接得上** ⇔ 没有阻断级问题 ✓。"""
        return not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {"violations": self.violations, "warnings": self.warnings,
                "propTimeline": self.propTimeline, "revealOrder": self.revealOrder,
                "deletable": self.deletable, "ok": self.ok, "counts": self.counts}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _violate(report: ContinuityReport, code: str, shot: str, message: str, **extra: Any) -> None:
    report.violations.append({"code": code, "shot": shot, "message": message, **extra})


def check_continuity(plan: Any) -> ContinuityReport:
    """体检一份**连续性计划** ✓ —— ``{locations, characters, props, clues, shots}``（全可选 ✓）。

    ``shots`` 里每镜的形状 ✓（缺项按"没声明"处理 ✓ —— 会不会报取决于该规则是否要求声明 ✓）::

        {"shot_id": "kf001", "location": "L1", "characters": ["R5"],
         "prop_states": {"P1": "floor"}, "screen_direction": {"R5": "left"},
         "actions": [{"actor": "R5", "action": "睁眼", "prop": "P1", "effects": ["information"]}],
         "clue_reveals": ["C1"], "transition": {"motive": "action", "from": "kf000"}}
    """
    report = ContinuityReport()
    if not isinstance(plan, dict):
        _violate(report, "plan-shape", "", "计划必须是 JSON 对象 ✗")
        return report

    # ⚠️ 跨镜方向记忆是**模块级**的（要跨 shot 比 ✓）⇒ 每次体检必须先清零 ✗：
    #    初版忘了这一步 ⇒ **上一位调用者的方向会污染下一位** ✗（服务端长期运行 ⇒
    #    不同集/不同剧之间互相"越轴" ✗ —— 自检 ㉒ 专门钉这条 ✓）。
    _reset_direction_memory()

    tables = {name: {str(item.get("id")): item for item in _as_list(plan.get(name))
                     if isinstance(item, dict) and item.get("id")}
              for name in TABLE_FIELDS}
    shots = [item for item in _as_list(plan.get("shots")) if isinstance(item, dict)]

    _check_tables(report, tables)
    if not shots:
        report.warnings.append({"code": "no-shots", "message": "没有 shots ⇒ 无从判断连续性 ✓"})
        report.counts = {"shots": 0, "violations": 0, "warnings": len(report.warnings)}
        return report

    order = [str(shot.get("shot_id") or f"#{index + 1}") for index, shot in enumerate(shots)]
    position = {shot_id: index for index, shot_id in enumerate(order)}

    _check_prop_timeline(report, tables, shots, order)
    _check_screen_direction(report, tables, shots, order)
    _check_clue_reveal(report, tables, shots, order, position)
    _check_transitions(report, shots, order)
    _check_actions(report, tables, shots, order)

    report.counts = {"shots": len(shots), "violations": len(report.violations),
                     "warnings": len(report.warnings), "deletable": len(report.deletable)}
    return report


# ── 表级必填（§6-§9 ✓）────────────────────────────────────────────────────
def _check_tables(report: ContinuityReport, tables: dict[str, dict[str, dict[str, Any]]]) -> None:
    for name, required in TABLE_FIELDS.items():
        for key, item in tables[name].items():
            missing = [field_name for field_name in required
                       if not str(item.get(field_name) or "").strip()]
            if missing:
                _violate(report, "table-field", "",
                         f"{name}[{key}] 缺少必填字段 {missing} ✗"
                         f"（§6-§9 的字段就是给视频模型照抄的 ✓ 缺一项就得多猜一项 ✗）",
                         table=name, asset=key, missing=missing)


# ── §8 道具状态表（⭐ 本模块的核心 ✓）────────────────────────────────────
def _check_prop_timeline(report: ContinuityReport, tables: dict[str, dict[str, dict[str, Any]]],
                         shots: list[dict[str, Any]], order: list[str]) -> None:
    """道具状态**像时间线** ✓：状态变了 ⇒ **同一镜里必须有交代它的动作** ✓。

    契约原话 ✓：*刀掉到地上，就不能再回到角色手里 —— 除非有一个镜头展示了那个人把它捡起来* ✓
    ⇒ 所以判据是「**变**」而不是「有」✓：一直在地上的刀不需要每镜都交代 ✓（那才是好的 ✓）。
    """
    timeline: dict[str, list[list[str]]] = {}
    previous: dict[str, str] = {}
    for shot_id, shot in zip(order, shots):
        states = shot.get("prop_states")
        if not isinstance(states, dict):
            continue
        actions = [item for item in _as_list(shot.get("actions")) if isinstance(item, dict)]
        for prop_id, state in states.items():
            key, value = str(prop_id), str(state)
            timeline.setdefault(key, []).append([shot_id, value])
            before = previous.get(key)
            previous[key] = value
            if before is None or before == value:
                continue
            # 状态**变了** ⇒ 必须有一个动作「碰过它」✓
            touching = [item for item in actions
                        if str(item.get("prop") or "") == key
                        or key in [str(other) for other in _as_list(item.get("props"))]]
            if not touching:
                _violate(report, "prop-timeline", shot_id,
                         f"道具 {key} 的状态从 {before} 变成 {value} ✗ 但**这一镜没有动作交代它** ✓"
                         f"（§8：状态变化必须有镜头说明是谁、怎么弄的 ✗ —— "
                         f"否则就是「刀自己飞回手里」✓）",
                         prop=key, frm=before, to=value)
            elif not str(touching[0].get("actor") or "").strip():
                _violate(report, "prop-timeline-actor", shot_id,
                         f"道具 {key} 状态变化有动作，但动作**没写谁做的** ✗（§8 要求「谁接触过它」✓）",
                         prop=key)
    report.propTimeline = timeline


# ── §6 场景空间表（画面方向不许翻 ✓）────────────────────────────────────
def _check_screen_direction(report: ContinuityReport,
                            tables: dict[str, dict[str, dict[str, Any]]],
                            shots: list[dict[str, Any]], order: list[str]) -> None:
    """同场景内同一角色的**画面方向不许翻转** ✓（越轴 ✗）—— 除非两镜之间有转场动机 ✓。"""
    for shot_id, shot in zip(order, shots):
        location = str(shot.get("location") or "")
        if location and location not in tables["locations"]:
            _violate(report, "unknown-location", shot_id,
                     f"镜头在未登记的场景 {location} ✗（§6 说场景要建表 ✓ 否则空间关系无从保证 ✓）")
        direction = shot.get("screen_direction")
        if not isinstance(direction, dict):
            continue
        for role_id, side in direction.items():
            key = str(role_id)
            entry = _direction_memory.setdefault(key, {})
            wanted = str(side).strip().lower()
            previous = entry.get(location)
            if previous is not None and previous != wanted and not _has_motive(shot):
                _violate(report, "axis-jump", shot_id,
                         f"角色 {key} 在场景 {location} 内画面方向从 {previous} 翻到 {wanted} ✗"
                         f"（**越轴** ✓）—— 除非这一镜给了转场动机 ✓",
                         character=key, frm=previous, to=wanted)
            entry[location] = wanted


#: ``_check_screen_direction`` 的跨镜记忆 ✓（逐次 ``check_continuity`` 调用独立 ✓ —— 每次清零 ✓）
_direction_memory: dict[str, dict[str, str]] = {}


# ── §9 线索 / 信息揭示表（不许提前暴露 ✓）──────────────────────────────
def _check_clue_reveal(report: ContinuityReport,
                       tables: dict[str, dict[str, dict[str, Any]]],
                       shots: list[dict[str, Any]], order: list[str],
                       position: dict[str, int]) -> None:
    revealed: list[list[str]] = []
    seen: set[str] = set()
    for shot_id, shot in zip(order, shots):
        shown = shot.get("clues_visible")
        if not isinstance(shown, (list, tuple)):
            continue
        for clue_id in shown:
            key = str(clue_id)
            entry = tables["clues"].get(key)
            if entry is None:
                _violate(report, "unknown-clue", shot_id,
                         f"镜头出现未登记的线索 {key} ✗（§9 要求线索先建表 ✓ 否则没法管揭示顺序 ✓）")
                continue
            declared = str(entry.get("first_visible_shot") or "")
            declared_index = position.get(declared)
            if declared_index is None and declared and declared != shot_id:
                # ⚠️ 表里声明的镜头**不在本次计划**里 ⇒ 先后**无法比较** ✓ ⇒ 只能预警 ✓
                #    （初版直接跳过 ✗ ⇒ 一条"提前暴露"被静默漏掉 ✓；也不该硬报阻断 ✗ ——
                #     因为我们确实不知道那一镜排在哪 ✓。**判不了的就说判不了** ✓。）
                report.warnings.append({
                    "code": "clue-declared-absent", "shot": shot_id, "clue": key,
                    "message": f"线索 {key} 表里声明首次可见于 {declared} ✗ 但本次计划里没有那一镜 ✓"
                               f"——它在 {shot_id} 就出现了 ✓ 先后无法比较 ✓（请人工核对 ✓）"})
            elif declared_index is not None and position.get(shot_id, 0) < declared_index:
                _violate(report, "clue-early", shot_id,
                         f"线索 {key} 在 {shot_id} 就露了 ✗ 但表里写的首次可见是 {declared} ✓"
                         f"（§9：**禁止提前暴露** ✓ —— 提前露了后面的悬念就没了 ✗）",
                         clue=key, declared=declared)
            if key not in seen:
                seen.add(key)
                revealed.append([shot_id, key])
    report.revealOrder = revealed

    # 表里声明了「首次可见」的线索，若整份计划里从未出现 ⇒ 报（图省事的典型遗漏 ✓）
    for key, entry in tables["clues"].items():
        if key not in seen:
            report.warnings.append({"code": "clue-missing", "message":
                                    f"线索 {key} 在表里声明了（首次可见 "
                                    f"{entry.get('first_visible_shot')} ✓）但**任何镜头都没让它出现** ✗"})


def _has_motive(shot: dict[str, Any]) -> bool:
    transition = shot.get("transition")
    if not isinstance(transition, dict):
        return False
    return str(transition.get("motive") or "").strip() in TRANSITION_MOTIVES


# ── §11 转场动机表（不许只写「切到下一镜」✓）───────────────────────────
def _check_transitions(report: ContinuityReport, shots: list[dict[str, Any]],
                       order: list[str]) -> None:
    for index, (shot_id, shot) in enumerate(zip(order, shots)):
        if index == 0:
            continue                     # 第一镜没有"上一镜"⇒ 不需要转场动机 ✓
        transition = shot.get("transition")
        if not isinstance(transition, dict):
            _violate(report, "transition-missing", shot_id,
                     "这一镜没有 transition ✗（§11：每一次切都要说清**为什么切** ✓）")
            continue
        motive = str(transition.get("motive") or "").strip()
        if motive.lower() in _EMPTY_MOTIVES:
            _violate(report, "transition-empty", shot_id,
                     f"转场动机写的是「{motive or '（空）'}」✗ ⇒ 等于没写 ✓"
                     f"（§11 原话：**永远不要只写「切到下一镜」** ✓ 要从八类里点名 ✓）",
                     motives=list(TRANSITION_MOTIVES))
        elif motive not in TRANSITION_MOTIVES:
            _violate(report, "transition-unknown", shot_id,
                     f"转场动机 {motive!r} 不在八类里 ✗（可用：{list(TRANSITION_MOTIVES)} ✓）")
        if str(transition.get("from") or "").strip() and index > 0:
            previous_shot = order[index - 1]
            if str(transition["from"]) != previous_shot:
                report.warnings.append({"code": "transition-source", "shot": shot_id,
                                        "message": f"transition.from={transition['from']} "
                                                   f"但上一镜其实是 {previous_shot} ✓（顺序对不上 ✓）"})


# ── §10 动作因果表（动作必须"做成点什么" ✓）─────────────────────────────
def _check_actions(report: ContinuityReport, tables: dict[str, dict[str, dict[str, Any]]],
                   shots: list[dict[str, Any]], order: list[str]) -> None:
    for shot_id, shot in zip(order, shots):
        actions = [item for item in _as_list(shot.get("actions")) if isinstance(item, dict)]
        if not actions:
            _violate(report, "action-missing", shot_id,
                     "这一镜没有任何 action ✗（每个镜头必须**只做一件事** ✓ 但也不能什么都不做 ✗）")
            continue
        if len(actions) > 1:
            report.warnings.append({"code": "action-multiple", "shot": shot_id,
                                    "message": f"这一镜有 {len(actions)} 个动作 ✓ 参考项目要求"
                                               f"「每个镜头只做一件事」✓ 建议拆镜或标注主次 ✓"})
        cast = {str(item) for item in _as_list(shot.get("characters"))}
        for action in actions:
            actor = str(action.get("actor") or "").strip()
            if not actor:
                _violate(report, "action-actor", shot_id,
                         "动作没写 actor ✗（§10 要「谁做了什么」✓）")
            elif cast and actor not in cast:
                _violate(report, "action-actor-absent", shot_id,
                         f"动作的 actor={actor} **不在这一镜的角色里** ✗"
                         f"（在场角色：{sorted(cast)} ✓ —— 这是最常见的接戏事故 ✓）",
                         actor=actor, cast=sorted(cast))
            elif actor and actor not in tables["characters"]:
                report.warnings.append({"code": "actor-unregistered", "shot": shot_id,
                                        "message": f"actor={actor} 不在角色表里 ✓"
                                                   f"（§7 的角色连续性卡是给视频模型照抄的 ✓）"})
            effects = [str(item) for item in _as_list(action.get("effects"))]
            valid = [item for item in effects if item in ACTION_EFFECTS]
            if not valid:
                detail = {
                    "shot": shot_id, "code": "action-inert", "actor": actor,
                    "action": str(action.get("action") or ""),
                    "message": f"{shot_id} 的动作「{action.get('action') or actor}」**一件都没做到** ✗"
                               f"（§10 原话：不推进剧情/揭示信息/改变危险/建立空间/改变情绪的动作，"
                               f"就标为**可删或可并** ✓）",
                }
                report.deletable.append(detail)
                report.warnings.append({**detail, "effects": list(ACTION_EFFECTS)})


def _reset_direction_memory() -> None:
    """清空跨镜方向记忆 ✓（自检用 ✓ —— 每次体检应当是**独立**的 ✓）。"""
    _direction_memory.clear()
