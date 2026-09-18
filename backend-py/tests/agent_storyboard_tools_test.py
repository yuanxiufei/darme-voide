"""S5 工具层自检：分镜拆解工具集（``agents/tools/storyboard_tools.py``）。四个工具。

四块**错了不报错、只是结果不对**的逻辑：

1. **``read_storyboard_context`` 的过滤规则**：角色/场景是「本集已关联的优先，
   **未关联任何东西时全给**」（空关联集 = 不过滤）；且原文取值链是 ``scriptContent || content``；
2. **``save_storyboards`` 的整集重建**：先删旧分镜**及其角色关联**（物品关联不删）→
   逐镜校验绑定 → **说话人自动绑定**（对白首个「角色名：」→ 角色名到 speaker_id 映射）→
   **对白一致性校验**（只告警不改数据，旁白/画外音跳过）→ 集时长 = ``ceil(总秒/60)`` → 三件后置；
3. **``save_continuity_states`` 是整体替换**（幂等：再存一次不会累积）；
4. **``update_storyboard`` 按「字段是否出现」写** ⇒ 传 ``null`` **会真的清空**该列，
   而不传则保持原值。

运行::

    ./.venv/Scripts/python.exe tests/agent_storyboard_tools_test.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="sbtools_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.core.models import (
    characters,
    continuity_states,
    episode_characters,
    episode_scenes,
    episodes,
    scenes,
    storyboard_characters,
    storyboards,
)  # noqa: E402
from app.core.response import now  # noqa: E402
from app.agent.tools import storyboard_tools as st  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _call(tools: dict, tool_id: str, arguments: dict | None = None):
    return asyncio.run(tools[tool_id].run(arguments))


_STAMP_CALLS: list[int] = []
_TAKE_CALLS: list[int] = []


def main() -> int:  # noqa: C901
    # 后置调用换成记录器（真身会写库）
    st.stamp_storyboards_script_hash = lambda conn, ep: _STAMP_CALLS.append(ep)  # type: ignore[assignment]
    st.reset_take_budget_for_episode = lambda conn, ep: _TAKE_CALLS.append(ep)  # type: ignore[assignment]

    client = TestClient(app)
    drama_id = client.post("/api/v1/dramas", json={"title": "SMOKE 分镜工具"}).json()["data"]["id"]
    episode_id = client.get(f"/api/v1/dramas/{drama_id}").json()["data"]["episodes"][0]["id"]

    scene_id = client.post("/api/v1/scenes", json={
        "drama_id": drama_id, "location": "客栈", "time": "夜晚"}).json()["data"]["id"]
    char1 = _insert_character(drama_id, "林昭", speaker_id="S1", appearance="青衫短发")
    char2 = _insert_character(drama_id, "阿晚", speaker_id="S2")

    tools = st.create_storyboard_tools(episode_id, drama_id)
    check("工具: 四个工具，键即 id",
          set(tools) == {"read_storyboard_context", "save_storyboards",
                         "update_storyboard", "save_continuity_states"}
          and all(k == t.id for k, t in tools.items()), sorted(tools))
    check("schema: save_storyboards 只有 storyboards 必填，逐镜只有 shot_number 必填",
          tools["save_storyboards"].input_schema["required"] == ["storyboards"]
          and tools["save_storyboards"].input_schema["properties"]["storyboards"]["items"]
          ["required"] == ["shot_number"])
    check("schema: update_storyboard / save_continuity_states 的必填项",
          tools["update_storyboard"].input_schema["required"] == ["storyboard_id"]
          and tools["save_continuity_states"].input_schema["properties"]["states"]["items"]
          ["required"] == ["state_type", "entity_key", "state_value"])

    # ================= 读取上下文 =================
    check("读取: 剧集不存在 -> Episode not found",
          _call(st.create_storyboard_tools(999999, drama_id), "read_storyboard_context")
          == {"error": "Episode not found"})
    check("读取: 无原文 -> Episode has no script",
          _call(tools, "read_storyboard_context") == {"error": "Episode has no script"})

    with engine.begin() as conn:
        conn.execute(episodes.update().where(episodes.c.id == episode_id)
                     .values(script_content="剧本正文", content="旧正文"))

    ctx = _call(tools, "read_storyboard_context")
    check("读取: 顶层字段齐全（含 script/style_id/episode/五组数据）",
          set(ctx) == {"style_id", "episode", "script", "script_truncated",
                       "script_total_chars", "characters", "scenes",
                       "existing_storyboards", "continuity_states", "props"},
          sorted(ctx))
    check("读取: 取值链是 **scriptContent 优先**（不是 content）",
          ctx["script"] == "剧本正文", ctx["script"])
    check("读取: episode 摘要字段",
          ctx["episode"]["episode_number"] == 1 and ctx["episode"]["title"] is not None,
          ctx["episode"])
    # ⭐ 空关联集 = 不过滤
    check("读取: 本集**没有**关联任何角色/场景时 -> 全给（空集视为不过滤）",
          sorted(c["name"] for c in ctx["characters"]) == ["林昭", "阿晚"]
          and len(ctx["scenes"]) == 1,
          ([c["name"] for c in ctx["characters"]], len(ctx["scenes"])))
    check("读取: appearance_hint = `名字: 外观||描述||定位||''`",
          [c["appearance_hint"] for c in ctx["characters"] if c["name"] == "林昭"] == ["林昭: 青衫短发"],
          ctx["characters"][0])

    # 建立本集关联后 -> 只给已关联的
    with engine.begin() as conn:
        conn.execute(episode_characters.insert().values(
            episode_id=episode_id, character_id=char1, created_at=now()))
        conn.execute(episode_scenes.insert().values(
            episode_id=episode_id, scene_id=scene_id, created_at=now()))
    ctx2 = _call(tools, "read_storyboard_context")
    check("读取: 有关联后**只给已关联**的角色/场景",
          [c["name"] for c in ctx2["characters"]] == ["林昭"] and len(ctx2["scenes"]) == 1,
          [c["name"] for c in ctx2["characters"]])

    # ================= 保存分镜（整集重建 + 说话人绑定 + 对白校验）=================
    _STAMP_CALLS.clear()
    _TAKE_CALLS.clear()
    saved = _call(tools, "save_storyboards", {"storyboards": [
        {"shot_number": 1, "title": "开场", "duration": 12, "scene_id": scene_id,
         "character_ids": [char1], "dialogue": "林昭：你来了，好久不见"},
        {"shot_number": 2, "title": "反应", "duration": 8, "scene_id": scene_id,
         "character_ids": [char1], "dialogue": "张三：路人甲的台词不该出现"},
        {"shot_number": 3, "title": "空镜", "scene_id": scene_id},
    ]})
    with engine.begin() as conn:
        rows = conn.execute(select(storyboards)
                            .where(storyboards.c.episode_id == episode_id)
                            .order_by(storyboards.c.storyboard_number)).all()
        ep_row = conn.execute(select(episodes).where(episodes.c.id == episode_id)).first()
    check("保存: 三镜落库、按镜号排序、返回 count/total_duration",
          len(rows) == 3 and saved["count"] == 3 and saved["total_duration"] == 30,
          (len(rows), saved))
    check("保存: **时长缺省是 10**（第三镜没给）",
          rows[2].duration == 10, rows[2].duration)
    check("保存: 集时长 = ceil(总秒/60)（30s -> 1 分钟）",
          ep_row.duration == 1, ep_row.duration)
    check("保存: 说话人自动绑定（对白「林昭：」-> S1）",
          rows[0].speaker_id == "S1", rows[0].speaker_id)
    check("保存: 对白里未知角色计入 dialogue_issues 并在回执里提示",
          saved["dialogue_issues"] == 1
          and "1 dialogue character mismatches detected" in saved["message"],
          (saved["dialogue_issues"], saved["message"]))
    check("保存: 无对白的镜头不参与校验（issues 只算 1 条）",
          rows[2].speaker_id == "", rows[2].speaker_id)
    with engine.begin() as conn:
        links = conn.execute(select(storyboard_characters.c.character_id)
                             .where(storyboard_characters.c.storyboard_id == rows[0].id)).all()
    check("保存: 角色关联已写入（第 1 镜 -> 林昭）", [x[0] for x in links] == [char1], links)
    check("保存: 三件后置 —— 剧本指纹盖章 + take 预算重置都被调用",
          _STAMP_CALLS == [episode_id] and _TAKE_CALLS == [episode_id],
          (_STAMP_CALLS, _TAKE_CALLS))

    # 重建：再存一次 -> 旧分镜（含关联）被清掉
    _call(tools, "save_storyboards", {"storyboards": [
        {"shot_number": 1, "title": "新开场", "duration": 60, "scene_id": scene_id}]})
    with engine.begin() as conn:
        after = conn.execute(select(storyboards)
                             .where(storyboards.c.episode_id == episode_id)).all()
        orphan_links = conn.execute(select(storyboard_characters.c.storyboard_id)).all()
    check("保存: **整集重建** —— 旧分镜被删光，只剩新的 1 镜",
          len(after) == 1 and after[0].title == "新开场", [r.title for r in after])
    check("保存: 旧分镜的**角色关联一并清掉**（不留孤儿关联）",
          all(x[0] == after[0].id for x in orphan_links), orphan_links)
    with engine.begin() as conn:
        ep2 = conn.execute(select(episodes).where(episodes.c.id == episode_id)).first()
    check("保存: 集时长随重建刷新（60s -> 1 分钟）", ep2.duration == 1, ep2.duration)

    # 绑定校验
    err = ""
    try:
        _call(tools, "save_storyboards", {"storyboards": [{
            "shot_number": 1, "duration": 10, "scene_id": 999999}]})
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
    # ⚠️ 此处用的是**共享 helper** `storyboard_helpers.validate_storyboard_bindings`，
    #    其文案是「scene_id 必须来自当前集已关联场景」（TS 原文是 `scene_id N 不属于当前集`，
    #    helper 少了那个 id）。差异已记录，不在本轮擅改共享件。
    check("保存: scene_id 不属于本集 -> 抛错",
          "scene_id" in err and "当前集" in err, err)

    # ================= 连续性状态 =================
    first = _call(tools, "save_continuity_states", {"states": [
        {"state_type": "scene_space", "entity_key": "客栈", "state_value": "门口朝内"},
        {"state_type": "prop_state", "entity_key": "玉坠", "state_value": "林昭持有",
         "storyboard_id": None, "constraints": "不落地"},
    ]})
    # ⚠️ 2026-09-18 **有意扩展**收据 ✓（原断言是严格 `== {message, count}` ✓）：
    #    现在另带 `problems` / `allowedStateTypes` / `vocabulary` 等**反馈字段** ✓ ——
    #    那是为了让模型下一轮能自己改对 ✓（工具返回就是它的反馈通道 ✓）；
    #    **`message` 与 `count` 逐字保持不变** ✓（外部依赖的就是这两个 ✓）。
    check("状态: 回执是 `Saved N continuity states` + count（逐字不变 ✓）",
          first["message"] == "Saved 2 continuity states" and first["count"] == 2, first)
    check("状态: ⭐ 回执里带**词汇表**与 `problems`（模型据此自纠 ✓）",
          bool(first["allowedStateTypes"]) and first["problems"] == []
          and first["vocabulary"] == [], first.get("allowedStateTypes"))
    with engine.begin() as conn:
        state_rows = conn.execute(select(continuity_states)
                                  .where(continuity_states.c.episode_id == episode_id)
                                  .order_by(continuity_states.c.id)).all()
    check("状态: 落库字段（含 null 的 storyboard_id 与默认空 constraints）",
          len(state_rows) == 2 and state_rows[0].constraints == ""
          and state_rows[1].storyboard_id is None,
          [(r.state_type, r.constraints, r.storyboard_id) for r in state_rows])

    _call(tools, "save_continuity_states", {"states": [
        {"state_type": "character_pose", "entity_key": "苏婉", "state_value": "站在吧台内侧"}]})
    with engine.begin() as conn:
        replaced = conn.execute(select(continuity_states)
                                .where(continuity_states.c.episode_id == episode_id)).all()
    check("状态: **整体替换**（再存一次不会累积）",
          len(replaced) == 1 and replaced[0].entity_key == "苏婉",
          [r.entity_key for r in replaced])
    empty_state = _call(tools, "save_continuity_states", {"states": []})
    with engine.begin() as conn:
        left = conn.execute(select(continuity_states.c.id)
                            .where(continuity_states.c.episode_id == episode_id)).all()
    check("状态: 空数组 -> 全部清空且 count=0",
          empty_state["count"] == 0 and left == [], (empty_state, left))

    # ================= 单镜更新 =================
    target = _call(tools, "read_storyboard_context")["existing_storyboards"][0]["id"]
    check("更新: 分镜不存在 -> error",
          _call(tools, "update_storyboard", {"storyboard_id": 999999})
          == {"error": "Storyboard 999999 not found"})

    got = _call(tools, "update_storyboard", {
        "storyboard_id": target, "title": "改标题", "movement": "推镜"})
    with engine.begin() as conn:
        upd = conn.execute(select(storyboards).where(storyboards.c.id == target)).first()
    check("更新: 只写传入的字段，未传字段保持原值",
          got == {"message": f"Storyboard {target} updated"}
          and upd.title == "改标题" and upd.movement == "推镜"
          and upd.duration == 60,
          (upd.title, upd.movement, upd.duration))

    _call(tools, "update_storyboard", {"storyboard_id": target, "scene_type": None})
    with engine.begin() as conn:
        cleared = conn.execute(select(storyboards).where(storyboards.c.id == target)).first()
    check("更新: 显式传 null -> **真的清空该列**（字段出现即写）",
          cleared.scene_type is None and cleared.title == "改标题", cleared.scene_type)
    _call(tools, "update_storyboard", {"storyboard_id": target, "speaker_id": "S9"})
    with engine.begin() as conn:
        after_speaker = conn.execute(select(storyboards).where(storyboards.c.id == target)).first()
    check("更新: 不传 scene_type 时不会被误清（只有出现的字段才写）",
          after_speaker.scene_type is None and after_speaker.speaker_id == "S9",
          (after_speaker.scene_type, after_speaker.speaker_id))

    # 绑定校验回退：不传 scene_id 时用库里已有的（本镜属于本集，应当通过）
    ok_update = _call(tools, "update_storyboard", {"storyboard_id": target, "action": "开门"})
    check("更新: 未传 scene_id/character_ids 时用库里已有值做校验（通过）",
          "message" in ok_update, ok_update)

    # ================= 汇总 =================
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


def _insert_character(drama_id: int, name: str, **extra) -> int:
    values: dict[str, object] = {"drama_id": drama_id, "name": name}
    for column in ("created_at", "updated_at"):
        if column in characters.c:
            values[column] = now()
    values.update(extra)
    with engine.begin() as conn:
        return int(conn.execute(characters.insert().values(**values)).lastrowid)


if __name__ == "__main__":
    raise SystemExit(main())
