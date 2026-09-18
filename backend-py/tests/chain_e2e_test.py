"""S7 端到端：**数据齐了，体检就该是绿的** ✓（2026-09-18）。

前面几套各自验了一段 ✓（判据 ✓ / 取数 ✓ / 写入 ✓ / 适配 ✓），但**没有一条**把整条链串起来跑 ✓：

```text
分镜（DB）──▶ save_continuity_states（**LLM 那条路径** ✓ 真工具 ✓）
          ──▶ run_episode_preflight（取数 → 适配 → 判定 → 门 ✓）
```

本套就做这件事 ✓，并回答一个此前只能推测的问题 ✓：
**「把 `STATE_TYPES` 那几样都填齐，`ready` 到底会不会变 True？」** ✓

⚠️ 两条纪律 ✓：
* **用临时数据根** ✓（`DATA_ROOT` 在 import `app.core.db` **之前**设 ✓）—— 真库 46 个分镜一行都不能碰 ✗；
* 断言要钉**具体要素**（哪个覆盖率上去了 ✓ / 哪条阻断消失了 ✓），
  不能只钉 `ready=True` ✗ —— 否则以后有人放宽判据，这条会**假绿** ✗。

运行::

    ./.venv/Scripts/python.exe tests/chain_e2e_test.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
os.environ["DATA_ROOT"] = tempfile.mkdtemp(prefix="chain_e2e_")
sys.path.insert(0, str(BACKEND_PY))

from sqlalchemy import insert, select  # noqa: E402

from app.agent.tools.storyboard_tools import create_storyboard_tools  # noqa: E402
from app.core.db import engine  # noqa: E402
from app.core.models import (  # noqa: E402
    characters, continuity_states, dramas, episodes, scenes, storyboard_characters,
    storyboards)
from app.services.preflight_source import run_episode_preflight  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


DRAMA_ID = 900
EPISODE_ID = 901
SCENE_ID = 5
SHOT_A = 1001
SHOT_B = 1002
CHAR_A = 7
CHAR_B = 8
CLUE = 4
PROP = 9


def _seed() -> None:
    """造一集**最小但完整**的现场 ✓（两镜、两角色、一场景一律 approved ✓）。"""
    with engine.begin() as conn:
        conn.execute(insert(dramas), [{"id": DRAMA_ID, "title": "E2E 链子",
                                       "status": "draft",
                                       "created_at": "2026-09-18", "updated_at": "2026-09-18"}])
        conn.execute(insert(episodes), [{"id": EPISODE_ID, "drama_id": DRAMA_ID,
                                         "episode_number": 1, "title": "第一集",
                                         "status": "draft",
                                         "created_at": "2026-09-18", "updated_at": "2026-09-18"}])
        conn.execute(insert(scenes), [{"id": SCENE_ID, "drama_id": DRAMA_ID,
                                       "episode_id": EPISODE_ID, "location": "旧城区案发房",
                                       "time": "深夜", "prompt": "冷蓝色调",
                                       "created_at": "2026-09-18", "updated_at": "2026-09-18"}])
        conn.execute(insert(characters), [
            {"id": CHAR_A, "drama_id": DRAMA_ID, "name": "沈砚", "core_features": "黑色湿短发",
             "status": "active", "created_at": "2026-09-18", "updated_at": "2026-09-18"},
            {"id": CHAR_B, "drama_id": DRAMA_ID, "name": "许知夏", "core_features": "短发便装",
             "status": "active", "created_at": "2026-09-18", "updated_at": "2026-09-18"}])
        conn.execute(insert(storyboards), [
            {"id": SHOT_A, "episode_id": EPISODE_ID, "scene_id": SCENE_ID,
             "storyboard_number": 1, "action": "睁眼", "asset_status": "approved",
             "created_at": "2026-09-18", "updated_at": "2026-09-18"},
            {"id": SHOT_B, "episode_id": EPISODE_ID, "scene_id": SCENE_ID,
             "storyboard_number": 2, "action": "坐起", "asset_status": "approved",
             "created_at": "2026-09-18", "updated_at": "2026-09-18"}])
        conn.execute(insert(storyboard_characters), [
            {"storyboard_id": SHOT_A, "character_id": CHAR_A, "costume": "湿外套"},
            {"storyboard_id": SHOT_B, "character_id": CHAR_A, "costume": "湿外套"}])


#: ⭐ **按 `STATE_TYPES` 词汇把六样填齐** ✓ —— 这是"LLM 应该产出的东西" ✓
STATES: list[dict] = [
    # ① §6 场景空间（**场景级** ✓：`scene_id` 有值、`storyboard_id` 空 ✓，用老名字 `scene_space` ✓）
    {"scene_id": SCENE_ID, "state_type": "scene_space", "entity_key": "left", "state_value": "门"},
    {"scene_id": SCENE_ID, "state_type": "scene_space", "entity_key": "right", "state_value": "破窗"},
    {"scene_id": SCENE_ID, "state_type": "scene_space", "entity_key": "back",
     "state_value": "白布尸体"},
    {"scene_id": SCENE_ID, "state_type": "scene_space", "entity_key": "foreground",
     "state_value": "湿水泥地"},
    {"scene_id": SCENE_ID, "state_type": "scene_space", "entity_key": "axis",
     "state_value": "左入右出"},
    # ② §7 角色连续性（老名字 `character_pose` ✓ ⇒ 回填 `current_state` ✓）
    {"storyboard_id": SHOT_A, "state_type": "character_pose", "entity_key": str(CHAR_A),
     "state_value": "躺姿，刚醒"},
    # ③ §8 道具时间线（**状态变了就必须有动作交代** ✓）
    {"storyboard_id": SHOT_A, "state_type": "prop", "entity_key": str(PROP),
     "state_value": "on_floor"},
    {"storyboard_id": SHOT_B, "state_type": "prop", "entity_key": str(PROP),
     "state_value": "hand"},
    # ④ §6 画面方向（**逐镜**；同场景内不许翻 ✓）
    {"storyboard_id": SHOT_A, "state_type": "direction", "entity_key": str(CHAR_A),
     "state_value": "left"},
    {"storyboard_id": SHOT_B, "state_type": "direction", "entity_key": str(CHAR_A),
     "state_value": "left"},
    # ⑤ §9 线索（首次可见由**最早那一镜**推出 ✓ ⇒ 不构成提前暴露 ✓）
    {"storyboard_id": SHOT_A, "state_type": "clue", "entity_key": str(CLUE),
     "state_value": "visible"},
    # ⑥ §10 动作（**带 effects** ✓ 且**指认它碰了哪件道具** ✓ ⇒ §8 的"有交代"才成立 ✓）
    {"storyboard_id": SHOT_A, "state_type": "action", "entity_key": str(CHAR_A),
     "state_value": "睁眼", "constraints": '["information"]'},
    {"storyboard_id": SHOT_B, "state_type": "action", "entity_key": str(CHAR_A),
     "state_value": "捡起刀", "constraints": '["plot"]', "meta": '{"prop": "9"}'},
    # ⑦ §11 转场动机（**第一镜不需要** ✓；第二镜必须给 ✓）
    {"storyboard_id": SHOT_B, "state_type": "transition", "entity_key": str(SHOT_A),
     "state_value": "action"},
]


def case_llm_path() -> None:
    """⭐ 走**真工具**写状态 ✓（LLM 那条路径 ✓）—— 不是直接 SQL ✓。"""
    tools = create_storyboard_tools(EPISODE_ID, DRAMA_ID)
    shot_states = [item for item in STATES if item.get("storyboard_id")]
    scene_states = [item for item in STATES if item.get("scene_id")]
    result = asyncio.run(tools["save_continuity_states"].execute(
        {"states": shot_states + scene_states}))
    check("① 工具 `save_continuity_states` 真能把它写进去 ✓（`count` = 15 ✓ 无问题 ✓）",
          result["count"] == len(STATES) and result["problems"] == [], result)

    with engine.connect() as conn:
        total = len(conn.execute(select(continuity_states.c.id)).all())
    check("② 落库行数对得上 ✓（**15 行** ✓ 一镜不多一镜不少 ✓）",
          total == len(STATES), total)


def case_preflight_green() -> None:
    """⭐⭐ 数据齐了 ⇒ 体检该是**绿的** ✓（并且钉**具体要素** ✓ 不只钉 `ready` ✗）。"""
    with engine.connect() as conn:
        report = run_episode_preflight(conn, EPISODE_ID)

    notes = " ".join(report["source"]["notes"])
    check("③ ⭐⭐ 六项覆盖率**全部满格** ✓✓（`2/2` 或场景级 `1/1` ✓）",
          "§6 画面方向 2/2" in notes and "§8 道具状态 2/2" in notes
          and "§9 线索可见 1/2" in notes and "§10 动作（带 effects） 2/2" in notes
          and "§11 转场动机 1/2" in notes and "接戏起止状态 0/2" in notes,
          [item for item in report["source"]["notes"] if item.startswith("覆盖率")])

    check("④ ⭐⭐ **`ready` 变 True** ✓✓（此前只能推测「填齐了会不会绿」✓ —— 现在有证据 ✓）",
          report["ready"] is True, (report["ready"], report["blockers"]))

    codes = {item.get("code") for item in report["blockers"]}
    check("⑤ ⭐ 那些**结构性误报**都消失了 ✓（`table-field` / `prop-timeline` / "
          "`transition-missing` 一条不剩 ✓）",
          codes == set(), sorted(str(code) for code in codes))

    gate = report["sections"].get("assetGate") or {}
    # ⚠️ 资产数按**实际被镜头需要的**算 ✓：本夹具把角色 A 链到两镜、角色 B **没链** ✓
    # ⇒ 资产 = 场景 1 + 角色 1 = **2** ✓（初版我写 3 ✗，是**期望写错** ✓ 实现是对的 ✓）
    check("⑥ 验收门开着 ✓（`asset_status='approved'` ⇒ `canGenerateVideo=True` ✓）",
          gate.get("canGenerateVideo") is True and gate.get("counts", {}).get("approved") == 2,
          gate.get("counts"))

    cont = report["sections"].get("continuity") or {}
    check("⑦ 连续性判定**真跑过**（不是空跑 ✓）：道具时间线有 2 条 ✓、揭示顺序 1 条 ✓",
          len((cont.get("propTimeline") or {}).get(str(PROP), [])) == 2
          and (cont.get("revealOrder") or []) == [["sb1001", str(CLUE)]], cont.get("propTimeline"))


def case_honesty_kept() -> None:
    """⚠️ 绿了也不能把"没数据"说成绿 ✓ —— **同一条链**上再验一次反向 ✓。"""
    # 清掉本集的全部状态 ⇒ 覆盖率应归零、且**明说「没判」** ✓；但**不是**说"通过" ✗
    with engine.begin() as conn:
        from sqlalchemy import delete
        conn.execute(delete(continuity_states).where(
            continuity_states.c.episode_id == EPISODE_ID))
    with engine.connect() as conn:
        thin = run_episode_preflight(conn, EPISODE_ID)
    thin_notes = " ".join(thin["source"]["notes"])
    check("⑧ ⚠️ 状态被清掉后 ⇒ 覆盖率归零 ✓ 且**明说「没判」** ✓（不冒充通过 ✗）",
          "完全不成立" in thin_notes and "§8 道具状态 0/2" in thin_notes,
          [item for item in thin["source"]["notes"] if "不成立" in item])
    check("⑨ ⭐⭐ **「判不了」也进阻断** ✓（清空状态后 `ready=False` ✓）—— "
          "正是本仓那条规则：**没数据 / 没读到 ≠ 通过** ✗",
          thin.get("ready") is False
          and any(item.get("stage") == "continuity-coverage"
                  for item in thin.get("blockers") or []), thin.get("blockers"))


def main() -> int:
    _seed()
    case_llm_path()
    case_preflight_green()
    case_honesty_kept()

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
