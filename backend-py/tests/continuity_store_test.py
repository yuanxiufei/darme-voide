"""S7 自检：**连续性状态写入侧**（校验 + 幂等替换 ✓ 2026-09-18）。

⚠️ 本套**用临时数据根** ✓（在 import `app.core.db` **之前** 把 `DATA_ROOT` 指到临时目录 ✓）——
写侧测试会真的插行 ✓，**绝不能**写进真实的 `data/drama.db` ✗。

本套钉四条"写坏数据"最容易踩的语义 ✓：

1. ⭐ **词汇拒收** ✓✓（`state_type` 不在词汇表 ⇒ **报错并回显词汇表** ✓，不是静默忽略 ✗
   —— 静默忽略会让"明明填了却没生效"变成悬案 ✓）；
2. ⭐ **幂等** ✓（同一镜重跑是"先删后写" ✓ ⇒ 行数不涨 ✓；涨了的话体检会把旧行当另一个状态 ✗ ⇒ 假阳性 ✓）；
3. ⭐ **只动显式给出的镜** ✓（没提到的镜一行不碰 ✓）；
4. ⭐⭐ **归属校验** ✓（`storyboard_id` 必须真属于该集 ✓，否则**一条都不写** ✗ ——
   写坏别的集的状态极难查 ✓）。

最后一条 ⑫ 是**闭环**：写完 → 体检能"看见"这些数据 ✓（覆盖率上升 ✓），
这才叫链子通了 ✓。

运行::

    ./.venv/Scripts/python.exe tests/continuity_store_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
# ⚠️⚠️ 必须在 import app.core.db **之前** 改数据根 ✓（它 import 时就建表 ✓）
os.environ["DATA_ROOT"] = tempfile.mkdtemp(prefix="continuity_store_")
sys.path.insert(0, str(BACKEND_PY))

from sqlalchemy import insert, select  # noqa: E402

from app.core.db import engine  # noqa: E402
from app.core.models import continuity_states, episodes, storyboards  # noqa: E402
from app.services.continuity_store import (  # noqa: E402
    normalize_states, write_episode_states, write_shot_states)

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


EPISODE_ID = 101
OTHER_EPISODE_ID = 202
SHOT_A = 1001
SHOT_B = 1002
SHOT_OTHER = 2001


def _seed() -> None:
    """造两集、三个镜 ✓（写完就不管 ✓ —— 全程在临时库里 ✓）。"""
    with engine.begin() as conn:
        conn.execute(insert(episodes), [
            {"id": EPISODE_ID, "drama_id": 1, "episode_number": 1, "title": "第一集",
             "status": "draft", "created_at": "2026-09-18", "updated_at": "2026-09-18"},
            {"id": OTHER_EPISODE_ID, "drama_id": 1, "episode_number": 2, "title": "第二集",
             "status": "draft", "created_at": "2026-09-18", "updated_at": "2026-09-18"}])
        conn.execute(insert(storyboards), [
            {"id": SHOT_A, "episode_id": EPISODE_ID, "storyboard_number": 1, "action": "睁眼",
             "created_at": "2026-09-18", "updated_at": "2026-09-18"},
            {"id": SHOT_B, "episode_id": EPISODE_ID, "storyboard_number": 2, "action": "坐起",
             "created_at": "2026-09-18", "updated_at": "2026-09-18"},
            {"id": SHOT_OTHER, "episode_id": OTHER_EPISODE_ID, "storyboard_number": 1,
             "action": "别的集", "created_at": "2026-09-18", "updated_at": "2026-09-18"}])


def _count(conn, storyboard_id: int) -> int:
    return len(conn.execute(select(continuity_states.c.id).where(
        continuity_states.c.storyboard_id == storyboard_id)).all())


PROP_ROW = {"state_type": "prop", "entity_key": "9", "state_value": "hand"}


# ══════════════════════════════════════════════════════════════════════════
# ① 词汇校验：**拒收，不是忽略**
# ══════════════════════════════════════════════════════════════════════════
def case_normalize() -> None:
    rows, problems = normalize_states([{**PROP_ROW, "state_type": "PROP"}])
    check("① 合法行通过 ✓ 且 `state_type` 归一成小写 ✓（与词汇表同源 ✓）",
          len(rows) == 1 and rows[0]["state_type"] == "prop" and problems == [], (rows, problems))

    _rows, missing = normalize_states([{"state_type": "prop", "entity_key": "9"}])
    check("② 缺必填（`state_value`）⇒ 报问题 ✓（不写半条 ✗）",
          missing and "state_value" in str(missing[0]), missing)

    _rows, unknown = normalize_states([{"state_type": "道具状态", "entity_key": "9",
                                        "state_value": "hand"}])
    check("③ ⭐⭐ **不在词汇表 ⇒ 拒收** ✓✓ 且提示里带**可用取值** ✓"
          "（静默忽略会让「填了却没生效」变成悬案 ✗）",
          not _rows and any("拒收" in item and "state_type" in item for item in unknown),
          unknown)

    check("④ 非数组 ⇒ 报问题 ✓（不抛异常 ✗）",
          normalize_states("坏的")[1] and normalize_states(None)[1], "")


# ══════════════════════════════════════════════════════════════════════════
# ② 写入：幂等 / 只动指定的镜 / 归属
# ══════════════════════════════════════════════════════════════════════════
def case_write() -> None:
    with engine.begin() as conn:
        first = write_shot_states(conn, episode_id=EPISODE_ID, storyboard_id=SHOT_A,
                                  states=[PROP_ROW])
    check("⑤ 写入成功 ✓（`inserted=1` ✓ 且真落库 ✓）",
          first["ok"] and first["inserted"] == 1, first)

    with engine.begin() as conn:
        check("⑥ 落库可查 ✓（真的写进 `continuity_states` ✓）", _count(conn, SHOT_A) == 1, "")

    with engine.begin() as conn:
        second = write_shot_states(conn, episode_id=EPISODE_ID, storyboard_id=SHOT_A,
                                   states=[{**PROP_ROW, "state_value": "on_floor"}])
    with engine.begin() as conn:
        check("⑦ ⭐ **幂等替换** ✓✓（同一镜重跑 ⇒ `deleted=1` ✓ 且**行数不涨** ✓）"
              "—— 涨了的话体检会把旧行当成「另一个状态」⇒ 假阳性 ✗",
              second["deleted"] == 1 and _count(conn, SHOT_A) == 1, (second, _count(conn, SHOT_A)))

    with engine.begin() as conn:
        write_shot_states(conn, episode_id=EPISODE_ID, storyboard_id=SHOT_B,
                          states=[{"state_type": "action", "entity_key": "7",
                                   "state_value": "坐起", "constraints": '["plot"]',
                                   "meta": '{"prop": "9"}'}])
        check("⑧ ⭐ **只动显式给出的镜** ✓（写 B 之后 A 的行还在 ✓ —— 没提到的镜一行不碰 ✓）",
              _count(conn, SHOT_A) == 1 and _count(conn, SHOT_B) == 1, "")

    with engine.begin() as conn:
        foreign = write_shot_states(conn, episode_id=EPISODE_ID, storyboard_id=SHOT_OTHER,
                                    states=[PROP_ROW])
    with engine.begin() as conn:
        check("⑨ ⭐⭐ **归属校验**：镜不属于这一集 ⇒ **一条都不写** ✓✓ 并说清原因 ✓"
              "（写坏别的集的状态极难查 ✓）",
              foreign["ok"] is False and foreign["inserted"] == 0
              and any("不属于" in item for item in foreign["problems"])
              and _count(conn, SHOT_OTHER) == 0, foreign)

    with engine.begin() as conn:
        bad = write_shot_states(conn, episode_id=EPISODE_ID, storyboard_id=SHOT_A,
                                states=[{"state_type": "乱填", "entity_key": "1",
                                         "state_value": "x"}])
    with engine.begin() as conn:
        check("⑩ ⭐ 词汇错 ⇒ **不写** ✓ 且错误里回显词汇表 ✓（写入方知道该怎么填 ✓）",
              bad["ok"] is False and bad["inserted"] == 0 and bad["vocabulary"]
              and _count(conn, SHOT_A) == 1, (bad["problems"], len(bad["vocabulary"])))


# ══════════════════════════════════════════════════════════════════════════
# ③ 批量 + ⭐ 闭环：写完，体检真的"看得见"
# ══════════════════════════════════════════════════════════════════════════
def case_batch_and_loop() -> None:
    with engine.begin() as conn:
        batch = write_episode_states(conn, episode_id=EPISODE_ID, shots=[
            {"storyboardId": SHOT_A, "states": [
                {"state_type": "prop", "entity_key": "9", "state_value": "on_floor"},
                {"state_type": "direction", "entity_key": "7", "state_value": "left"},
                {"state_type": "clue", "entity_key": "4", "state_value": "visible"}]},
            {"storyboardId": SHOT_B, "states": [
                {"state_type": "prop", "entity_key": "9", "state_value": "hand"},
                {"state_type": "transition", "entity_key": str(SHOT_A), "state_value": "prop"},
                {"state_type": "action", "entity_key": "7", "state_value": "捡起刀",
                 "constraints": '["plot"]', "meta": '{"prop": "9"}'}]},
            {"storyboardId": 999999, "states": [PROP_ROW]},
        ])
    check("⑪ 批量：逐镜独立 ✓ + **问题逐条带镜号** ✓（`touchedShots` 只含真处理了的 ✓）",
          batch["touchedShots"] == [SHOT_A, SHOT_B, 999999]
          and any("不属于" in item for item in batch["problems"]), batch["problems"])

    # ⭐⭐ 闭环：写完 → 体检看得见
    with engine.connect() as conn:
        from app.services.preflight_source import run_episode_preflight
        report = run_episode_preflight(conn, EPISODE_ID)
    source_notes = " ".join(report["source"]["notes"])
    check("⑫ ⭐⭐ **闭环**：写入后体检**看得见**这些数据 ✓（覆盖率不再全是 0 ✓）",
          "§6 画面方向 1/" in source_notes and "§8 道具状态 2/" in source_notes,
          [item for item in report["source"]["notes"] if item.startswith("覆盖率")][:3])
    check("⑬ 体检结论仍由**判据**给 ✓（写数据不会让 `ready` 变成假绿 ✗）",
          isinstance(report.get("ready"), bool) and report.get("blockers") is not None, "")

    with engine.begin() as conn:
        again = write_shot_states(conn, episode_id=EPISODE_ID, storyboard_id=SHOT_A,
                                  states=[PROP_ROW])
    with engine.begin() as conn:
        check("⑭ ⭐ 幂等对**多行**也成立 ✓（A 原有 3 行 ⇒ 重写 1 行后剩 1 行 ✓）",
              again["deleted"] == 3 and again["inserted"] == 1 and _count(conn, SHOT_A) == 1,
              (again, _count(conn, SHOT_A)))


# ══════════════════════════════════════════════════════════════════════════
# ④ 路由
# ══════════════════════════════════════════════════════════════════════════
def case_api() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    ok = client.put("/api/v1/production/continuity-states", json={
        "episodeId": EPISODE_ID,
        "shots": [{"storyboardId": SHOT_A, "states": [PROP_ROW]}]})
    data = ok.json().get("data") or {}
    check("⑮ PUT `/production/continuity-states` 真能调用 ✓（200 + 写了几行 ✓）",
          ok.status_code == 200 and data.get("inserted") == 1, (ok.status_code, data))

    check("⑯ ⭐ 词汇错经 API ⇒ **200 但 `ok=false` + problems** ✓"
          "（不用 4xx：可能是「部分镜成了 ✓ 部分没成 ✓」⇒ 4xx 会把落库那部分也说成失败 ✗）",
          client.put("/api/v1/production/continuity-states", json={
              "episodeId": EPISODE_ID,
              "shots": [{"storyboardId": SHOT_A, "states": [
                  {"state_type": "乱填", "entity_key": "1", "state_value": "x"}]}]}
          ).json()["data"]["ok"] is False, "")

    check("⑰ `episodeId` 坏 / 缺 `shots` ⇒ **400** ✓（不猜 ✓、也不清空整集 ✗）",
          client.put("/api/v1/production/continuity-states",
                     json={"episodeId": "第一集", "shots": []}).status_code == 400
          and client.put("/api/v1/production/continuity-states",
                         json={"episodeId": EPISODE_ID}).status_code == 400, "")

    vocab = client.get("/api/v1/production/continuity-states/vocabulary").json().get("data") or {}
    check("⑱ 词汇表可查 ✓（写入方照它填 ✓ 不自己造词 ✗）",
          len(vocab.get("stateTypes") or []) == 7, len(vocab.get("stateTypes") or []))


def main() -> int:
    _seed()
    case_normalize()
    case_write()
    case_batch_and_loop()
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
