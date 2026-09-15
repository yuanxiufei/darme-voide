"""多集节奏相位 —— 移植 ``services/rhythm-phase.ts``（173 行，整域关闭）。

对齐参考项目 multi-episode rhythm phase：把每集分镜按**其在集内的时长位置**划到四类节奏相位，
跨集统计节奏分布，帮 ``storyboard_breaker`` 保持节奏一致性。

相位划分（按累计时长占比）：

* ``setup``       0%–25%   建置：引入人物/场景/冲突前提
* ``development`` 25%–70%  推进：矛盾升级/伏笔展开
* ``climax``      70%–90%  高潮：冲突爆发
* ``resolution``  90%–100% 收束：落点/钩子

⚠️ 五处保真点（错了不会报错、只会让相位分布悄悄变形）：

1. **累计占比用的是「进入本镜之前」的累计值**（``cumulative / total`` 先算再用），所以首镜恒为 ``setup``；
2. **总时长为 0 时按「序号占比」**：``(storyboardNumber - 1) / 分镜数`` —— 注意用的是
   **storyboard_number**（不是循环下标）⇒ 编号不连续/不从 1 起时占比会越界，照样按阈值落相位（照抄）；
3. **已存相位优先**：``rhythm_phase || phaseForPosition(0.5)`` ⇒ 没打过相位时**默认 ``development``**
   （0.5 落在 25%–70% 区间，不是 setup）；
4. **``climax_storyboards`` 只取前 5 条**（QC 重点抽查用）；
5. **``phase_balance`` 按分镜数加权**（不是按集平均），且只在「有分镜」时才产出四个相位键。

⚠️ 落库键是 **snake_case**（``episode_id`` / ``total_duration`` / ``phase_balance`` …）——
与原 TS 接口一致，前端直接吃这个形状，**不要改成 camelCase**。
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import select, update
from sqlalchemy.engine import Connection

from ..core.models import episodes, storyboards
from ..core.response import js_round

__all__ = [
    "RHYTHM_PHASE_LABELS",
    "RHYTHM_PHASES",
    "assign_rhythm_phases",
    "get_drama_rhythm",
    "get_episode_rhythm",
    "phase_for_position",
    "rhythm_guidance_for_episode",
]

RHYTHM_PHASES: tuple[str, ...] = ("setup", "development", "climax", "resolution")
RHYTHM_PHASE_LABELS: dict[str, str] = {
    "setup": "建置",
    "development": "推进",
    "climax": "高潮",
    "resolution": "收束",
}


def phase_for_position(ratio: float) -> str:
    """按累计时长占比返回相位。"""
    if ratio < 0.25:
        return "setup"
    if ratio < 0.70:
        return "development"
    if ratio < 0.90:
        return "climax"
    return "resolution"


def _episode_storyboards(conn: Connection, episode_id: Any) -> list[Any]:
    """取某集分镜（**按分镜序号升序** —— 相位判定依赖这个顺序）。"""
    return conn.execute(
        select(storyboards)
        .where(storyboards.c.episode_id == episode_id)
        .order_by(storyboards.c.storyboard_number)
    ).all()


def assign_rhythm_phases(conn: Connection, episode_id: Any) -> int:
    """为某集所有分镜分配节奏相位（按累计时长占比；无时长数据则按序号占比）→ 返回写入条数。"""
    rows = _episode_storyboards(conn, episode_id)
    if not rows:
        return 0

    total_duration = sum(row.duration or 0 for row in rows)
    cumulative = 0
    assigned = 0
    for row in rows:
        if total_duration > 0:
            ratio = cumulative / total_duration
            cumulative += row.duration or 0
        else:
            # ⚠️ 用的是 storyboard_number（不是循环下标），照抄
            ratio = (row.storyboard_number - 1) / len(rows)
        conn.execute(
            update(storyboards)
            .where(storyboards.c.id == row.id)
            .values(rhythm_phase=phase_for_position(ratio))
        )
        assigned += 1
    return assigned


def get_episode_rhythm(conn: Connection, episode_id: Any) -> dict[str, Any] | None:
    """统计某集节奏分布（剧集不存在返回 ``None``）。"""
    episode = conn.execute(select(episodes).where(episodes.c.id == episode_id)).first()
    if episode is None:
        return None
    rows = _episode_storyboards(conn, episode_id)

    phases: dict[str, int] = {}
    climax_storyboards: list[int] = []
    total_duration = 0
    for row in rows:
        total_duration += row.duration or 0
        # ⚠️ `||`：空串/None 都回退；回退值是 development（0.5 的相位），不是 setup
        phase = row.rhythm_phase or phase_for_position(0.5)
        phases[phase] = phases.get(phase, 0) + 1
        if phase == "climax":
            climax_storyboards.append(row.storyboard_number)

    return {
        "episode_id": episode_id,
        "episode_number": episode.episode_number,
        "title": episode.title,
        "total_duration": total_duration,
        "storyboard_count": len(rows),
        "phases": phases,
        "climax_storyboards": climax_storyboards[:5],
        "complete": all((phases.get(p) or 0) > 0 for p in RHYTHM_PHASES),
    }


def rhythm_guidance_for_episode(conn: Connection, episode_id: Any,
                                _drama_rhythm: dict[str, Any] | None = None) -> str:
    """生成跨集节奏引导文本（注入 ``storyboard_breaker``；剧集不存在返回空串）。

    ⚠️ **有意修掉原 TS 的无限递归**：TS 里 ``getDramaRhythm`` 调 ``rhythmGuidanceForEpisode``，
    后者又调 ``getDramaRhythm`` ⇒ 互相递归、没有记忆化 ⇒ Node 侧栈溢出（``RangeError``，
    被路由 catch 成 400），**该端点在有剧集的剧上根本不能用**。这里允许把**已算好的报告**
    传进来打断环（内容完全等价：用的就是同一份 ``episodes`` / ``phase_balance``）。
    """
    episode = conn.execute(select(episodes).where(episodes.c.id == episode_id)).first()
    if episode is None:
        return ""
    drama_rhythm = _drama_rhythm if _drama_rhythm is not None else get_drama_rhythm(conn, episode.drama_id)
    previous = sorted(
        (e for e in drama_rhythm["episodes"] if e["episode_number"] < episode.episode_number),
        key=lambda e: e["episode_number"], reverse=True,
    )
    prev = previous[0] if previous else None

    lines: list[str] = []
    lines.append("【多集节奏相位要求】")
    lines.append("请将本集分镜按节奏划分为四相：setup(建置,前25%) / development(推进,25%-70%) / "
                 "climax(高潮,70%-90%) / resolution(收束,末10%)。")

    if prev:
        prev_phases = "、".join(f"{RHYTHM_PHASE_LABELS[p]}{prev['phases'].get(p) or 0}镜"
                                for p in RHYTHM_PHASES)
        lines.append(f"上一集（第{prev['episode_number']}集）节奏分布：{prev_phases}，"
                     f"总时长{js_round(prev['total_duration'])}秒。请保持节奏基调一致。")

    avg = drama_rhythm["phase_balance"]
    if drama_rhythm["episodes"]:
        shares = "、".join(
            f"{RHYTHM_PHASE_LABELS[p]}占比{js_round((avg.get(p) or 0) * 100)}%"
            for p in RHYTHM_PHASES
        )
        lines.append(f"该剧各集平均节奏：{shares}。若本集为高潮集，可加大 climax 占比。")

    lines.append("若上一集以悬念/钩子收尾，本集开局需回应；若上一集是普通推进，本集结尾应埋下钩子。")
    return "\n".join(lines)


def get_drama_rhythm(conn: Connection, drama_id: Any) -> dict[str, Any]:
    """生成全剧节奏报告（剧不存在时返回全零的空报告 —— 原 TS 不报错）。"""
    episode_rows = conn.execute(
        select(episodes).where(episodes.c.drama_id == drama_id)
        .order_by(episodes.c.episode_number)
    ).all()

    episode_rhythms = [
        rhythm for rhythm in (get_episode_rhythm(conn, ep.id) for ep in episode_rows)
        if rhythm is not None
    ]

    # 平均占比（**按分镜数量加权**）
    total_storyboards = sum(e["storyboard_count"] for e in episode_rhythms)
    phase_balance: dict[str, float] = {}
    if total_storyboards > 0:
        for phase in RHYTHM_PHASES:
            count = sum(e["phases"].get(phase) or 0 for e in episode_rhythms)
            phase_balance[phase] = count / total_storyboards

    report: dict[str, Any] = {
        "drama_id": drama_id,
        "total_episodes": len(episode_rhythms),
        "total_duration": sum(e["total_duration"] for e in episode_rhythms),
        "episodes": episode_rhythms,
        "phase_balance": phase_balance,
    }
    # ⚠️ 把**已算好的报告**传进去打断 `getDramaRhythm ↔ rhythmGuidanceForEpisode` 的环
    #    （原 TS 在这条路径上无限递归 ⇒ 端点不可用；见 `rhythm_guidance_for_episode` docstring）
    report["guidance"] = (rhythm_guidance_for_episode(
        conn, episode_rows[-1].id, _drama_rhythm=report
    ) if episode_rows else "")
    return report
