"""S7 自检：多集节奏相位（``rhythm-phase.ts`` 173 行 → `rhythm_phase.py`）+ ``GET /dramas/{id}/rhythm``。

锁的是四条「错了也不报错、只会让分布悄悄变形」的规则：**累计占比先算再用**、**总时长为 0 时按
``storyboard_number`` 占比**（不是循环下标）、**已存相位优先且缺省是 development**、**balance 按分镜数加权**。

运行::

    ./.venv/Scripts/python.exe tests/rhythm_phase_test.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="rhythm_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.core.models import dramas, episodes, storyboards  # noqa: E402
from app.core.response import now  # noqa: E402
from app.services import rhythm_phase as rp  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _seed_storyboards(conn, episode_id: int, spec: list[tuple[int, float | None, str | None]]) -> None:
    """spec: [(storyboard_number, duration, rhythm_phase), ...]"""
    stamp = now()
    for number, duration, phase in spec:
        conn.execute(storyboards.insert().values(
            episode_id=episode_id, storyboard_number=number, title=f"镜{number}",
            duration=duration, rhythm_phase=phase, created_at=stamp, updated_at=stamp))


def main() -> int:  # noqa: C901
    # ── 相位阈值 ──
    check("阈值: 0/24.99% -> setup，25% -> development（**边界属于下一档**）",
          [rp.phase_for_position(x) for x in (0, 0.2499, 0.25)]
          == ["setup", "setup", "development"])
    check("阈值: 70% -> climax、90% -> resolution（含越界 >1 一律 resolution）",
          [rp.phase_for_position(x) for x in (0.6999, 0.70, 0.8999, 0.90, 1.5)]
          == ["development", "climax", "climax", "resolution", "resolution"])

    stamp = now()
    with engine.begin() as conn:
        drama_id = int(conn.execute(dramas.insert().values(
            title="节奏剧", created_at=stamp, updated_at=stamp)).lastrowid)
        ep1 = int(conn.execute(episodes.insert().values(
            drama_id=drama_id, episode_number=1, title="第一集", content="x",
            created_at=stamp, updated_at=stamp)).lastrowid)
        ep2 = int(conn.execute(episodes.insert().values(
            drama_id=drama_id, episode_number=2, title="第二集", content="x",
            created_at=stamp, updated_at=stamp)).lastrowid)
        empty_ep = int(conn.execute(episodes.insert().values(
            drama_id=drama_id, episode_number=3, title="空集", content="x",
            created_at=stamp, updated_at=stamp)).lastrowid)
        # ep1：4 镜 × 10s ⇒ 累计占比 0 / .25 / .5 / .75
        _seed_storyboards(conn, ep1, [(1, 10.0, None), (2, 10.0, None),
                                      (3, 10.0, None), (4, 10.0, None)])
        # ep2：编号不从 1 起（10/20）且时长为 0 ⇒ 走「序号占比」分支
        _seed_storyboards(conn, ep2, [(10, 0.0, None), (20, 0.0, None)])

    # ── 分配 ──
    with engine.begin() as conn:
        check("分配: 空集 -> 0 条", rp.assign_rhythm_phases(conn, empty_ep) == 0)
        check("分配: 4 镜 × 10s -> 0/.25/.5/.75 各镜", rp.assign_rhythm_phases(conn, ep1) == 4)
        got = [row.rhythm_phase for row in conn.execute(
            select(storyboards).where(storyboards.c.episode_id == ep1)
            .order_by(storyboards.c.storyboard_number)).all()]
        check("分配: 相位 = setup/development/development/climax（**首镜恒 setup**）",
              got == ["setup", "development", "development", "climax"], got)
        rp.assign_rhythm_phases(conn, ep2)
        got2 = [row.rhythm_phase for row in conn.execute(
            select(storyboards).where(storyboards.c.episode_id == ep2)
            .order_by(storyboards.c.storyboard_number)).all()]
        check("分配: 总时长为 0 时按 **storyboard_number** 占比（编号 10/20 ⇒ 两条都落 resolution）",
              got2 == ["resolution", "resolution"], got2)

    # ── 单集统计 ──
    with engine.begin() as conn:
        check("统计: 剧集不存在 -> None", rp.get_episode_rhythm(conn, 999999) is None)
        ep1_stats = rp.get_episode_rhythm(conn, ep1)
    check("统计: 形状（snake_case 键 + 集号/标题/时长/镜数）",
          list(ep1_stats) == ["episode_id", "episode_number", "title", "total_duration",
                              "storyboard_count", "phases", "climax_storyboards", "complete"],
          list(ep1_stats))
    check("统计: phases 计数与 climax 列表、complete（缺 resolution ⇒ false）",
          ep1_stats["phases"] == {"setup": 1, "development": 2, "climax": 1}
          and ep1_stats["climax_storyboards"] == [4]
          and ep1_stats["complete"] is False
          and ep1_stats["total_duration"] == 40.0 and ep1_stats["storyboard_count"] == 4,
          ep1_stats)

    with engine.begin() as conn:
        ep3 = empty_ep  # 复用变量名，保持可读
        check("统计: 已存相位优先，**缺省回退 development**（不是 setup）",
              (lambda: (
                  _seed_storyboards(conn, ep3, [(1, 5.0, None), (2, 5.0, "setup")]),
                  rp.get_episode_rhythm(conn, ep3)["phases"],
              ))()[-1] == {"development": 1, "setup": 1},
              rp.get_episode_rhythm(conn, ep3))
        _seed_storyboards(conn, ep3, [(3, 1.0, "climax"), (4, 1.0, "climax"), (5, 1.0, "climax"),
                                      (6, 1.0, "climax"), (7, 1.0, "climax"), (8, 1.0, "climax")])
        climax_list = rp.get_episode_rhythm(conn, ep3)["climax_storyboards"]
        check("统计: `climax_storyboards` 只取**前 5 条**（QC 抽查用）",
              len(climax_list) == 5, climax_list)

    # ── 跨集报告 ──
    with engine.begin() as conn:
        empty_report = rp.get_drama_rhythm(conn, 999999)
        check("报告: 剧不存在 -> 全零空报告（**不抛错**，路由才能回 200）",
              empty_report["total_episodes"] == 0 and empty_report["episodes"] == []
              and empty_report["phase_balance"] == {} and empty_report["total_duration"] == 0
              and empty_report["guidance"] == "", empty_report)
        report = rp.get_drama_rhythm(conn, drama_id)
    check("报告: 集按 episode_number 升序、total_episodes/总时长累加",
          [e["episode_number"] for e in report["episodes"]] == [1, 2, 3]
          and report["total_episodes"] == 3
          and report["total_duration"] == 40.0 + 0.0 + (5.0 + 5.0 + 6.0), report["total_duration"])
    # ⚠️ 总镜数 = ep1(4) + ep2(2) + ep3(2+6) = **14**；各相位 2/3/7/2
    check("报告: **phase_balance 按分镜数加权**（不是按集平均）",
          report["phase_balance"]["setup"] == 2 / 14
          and report["phase_balance"]["development"] == 3 / 14
          and report["phase_balance"]["climax"] == 7 / 14
          and report["phase_balance"]["resolution"] == 2 / 14,
          report["phase_balance"])
    check("报告: guidance 取自**最后一集**（第 3 集）⇒ 含「上一集（第2集）」而非第 1 集",
          "上一集（第2集）" in report["guidance"], report["guidance"][:160])

    # ── 引导文本 ──
    with engine.begin() as conn:
        check("引导: 剧集不存在 -> 空串", rp.rhythm_guidance_for_episode(conn, 999999) == "")
        guidance = rp.rhythm_guidance_for_episode(conn, ep1)
    check("引导: 四相说明 + 上下集衔接要求（固定文案）",
          guidance.startswith("【多集节奏相位要求】")
          and "setup(建置,前25%)" in guidance
          and guidance.endswith("本集结尾应埋下钩子。"))
    check("引导: 第一集没有「上一集」段，但有「各集平均节奏」段",
          "上一集（" not in guidance and "该剧各集平均节奏：" in guidance, guidance[:200])
    with engine.begin() as conn:
        second = rp.rhythm_guidance_for_episode(conn, ep2)
    check("引导: 第二集会引用第 1 集分布，时长按 `Math.round` 取整（40 秒）",
          "上一集（第1集）节奏分布：建置1镜、推进2镜、高潮1镜、收束0镜" in second
          and "总时长40秒" in second, second[:220])

    # ── 端点 ──
    client = TestClient(app)
    ok = client.get(f"/api/v1/dramas/{drama_id}/rhythm")
    check("端点: 200 + data 为报告（snake_case，direct 吃这个形状）",
          ok.status_code == 200 and ok.json()["code"] == 200
          and ok.json()["data"]["drama_id"] == drama_id
          and "phase_balance" in ok.json()["data"], ok.text[:140])
    check("端点: 非法 id -> 404 'Invalid drama id'",
          client.get("/api/v1/dramas/abc/rhythm").status_code == 404)
    missing = client.get("/api/v1/dramas/999999/rhythm")
    check("端点: 剧不存在 -> **200 空报告**（原 TS 不检查存在性，别加 404）",
          missing.status_code == 200 and missing.json()["data"]["total_episodes"] == 0,
          missing.text[:140])

    failed_items = [item for item in _RESULTS if not item[1]]
    for name, ok_, detail in _RESULTS:
        print(("PASS  " if ok_ else "FAIL  ") + name + ("" if ok_ else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed_items)}/{len(_RESULTS)} passed")
    return 1 if failed_items else 0


if __name__ == "__main__":
    raise SystemExit(main())
