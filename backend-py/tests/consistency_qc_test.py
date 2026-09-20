"""S7 自检：图像连续性 QC ``run_episode_consistency_qc`` + ``POST /episodes/{id}/consistency-qc``。

**用真实图片**（ffmpeg lavfi 现场造）+ 真实 dHash ⇒ 三个分支都有**确定性真值**：

===============================  ==========  ==========================================
图对                              相似度       命中分支
===============================  ==========  ==========================================
同图（testsrc2 vs 自己）           1.0         ok（``相似度 1`` —— 整数不带 .0）
testsrc2 vs smptehdbars          0.5625      info（``0.563`` —— half-up 进位）
smptehdbars vs 升坡              0.4219      warning
升坡 vs 降坡                      0.0         warning（``相似度 0``）
===============================  ==========  ==========================================

运行::

    ./.venv/Scripts/python.exe tests/consistency_qc_test.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="consist_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import asyncio  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from app.core.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.core.models import dramas, episodes, storyboards, video_quality_checks  # noqa: E402
from app.core.response import now  # noqa: E402
from app.services import consistency_qc  # noqa: E402
from app.services.consistency_qc import run_episode_consistency_qc  # noqa: E402
from app.services.file_storage import get_storage_root  # noqa: E402

_R: list[tuple[str, bool, object]] = []
_CALLS = {"hash": 0}


def check(name: str, condition: object, detail: object = "") -> None:
    _R.append((name, bool(condition), detail))


# ---------------------------------------------------------------- 造图
def _ffmpeg(dest: Path, args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-v", "error", *args, "-frames:v", "1", "-y", str(dest)],
                   check=True, capture_output=True)


def build_images() -> dict[str, str]:
    """在 storage root 下造 4 张图，返回「相对路径」（与生产里存库的形态一致）。"""
    root = Path(get_storage_root())
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    _ffmpeg(images / "a.png", ["-f", "lavfi", "-i", "testsrc2=s=72x64"])
    _ffmpeg(images / "b.png", ["-f", "lavfi", "-i", "smptehdbars=s=72x64"])
    _ffmpeg(images / "up.png", ["-f", "lavfi", "-i", "color=black:s=72x64",
                                "-vf", "geq=lum='255*X/W':cb=128:cr=128"])
    _ffmpeg(images / "down.png", ["-f", "lavfi", "-i", "color=black:s=72x64",
                                  "-vf", "geq=lum='255*(1-X/W)':cb=128:cr=128"])
    (images / "corrupt.png").write_bytes(b"not an image")
    return {k: f"images/{k}.png" for k in ("a", "b", "up", "down", "corrupt")}


def seed(conn, number: int, episode_id: int, stamp: str, *, image: str | None = None,
         scene_id: int | None = 7, deleted: bool = False, field: str = "composed_image") -> int:
    values = {
        "episode_id": episode_id, "storyboard_number": number, "title": f"镜{number}",
        "description": "d", "scene_id": scene_id, "created_at": stamp, "updated_at": stamp,
        "deleted_at": stamp if deleted else None,
    }
    if image is not None:
        values[field] = image
    return int(conn.execute(storyboards.insert().values(**values)).lastrowid)


def make_episode(conn, pid: int, stamp: str, deleted: bool = False) -> int:
    return int(conn.execute(episodes.insert().values(
        drama_id=pid, episode_number=1, title="第一集", content="x",
        created_at=stamp, updated_at=stamp,
        deleted_at=stamp if deleted else None)).lastrowid)


def main() -> int:  # noqa: C901
    img = build_images()
    stamp = now()
    inner = consistency_qc._compute_dhash

    async def counting(path: str):
        _CALLS["hash"] += 1
        return await inner(path)

    consistency_qc._compute_dhash = counting  # type: ignore[assignment]

    with engine.begin() as conn:
        did = int(conn.execute(dramas.insert().values(
            title="连续性剧", created_at=stamp, updated_at=stamp)).lastrowid)

        # ── 场景 S1：同场景（scene 7）五镜，覆盖 ok / info / warning 全分支 ──
        e1 = make_episode(conn, did, stamp)
        s1 = [seed(conn, 1, e1, stamp, image=img["a"], field="composed_image"),
              seed(conn, 2, e1, stamp, image=img["a"], field="first_frame_image"),
              seed(conn, 3, e1, stamp, image=img["b"], field="keyframe_image"),
              seed(conn, 4, e1, stamp, image=img["up"]),
              seed(conn, 5, e1, stamp, image=img["down"])]
        # 2 号先塞一条**更旧**的 QC（先插 ⇒ id 更小）⇒ 服务必须只更新 id 最大那条
        old_qc_id = int(conn.execute(video_quality_checks.insert().values(
            storyboard_id=s1[1], status="failed", dimensions=json.dumps({"older": True}),
            issues=json.dumps([]), created_at=stamp, updated_at=stamp)).lastrowid)
        # 给 2/3/4/5 各一条 QC 记录（1 号**没有** ⇒ 覆盖「只写有 QC 的分镜」）
        for sb_id, dims, issues in ((s1[1], {"old": 1}, [{"dimension": "x", "message": "y"}]),
                                    (s1[2], {}, []), (s1[3], {}, []), (s1[4], {}, [])):
            conn.execute(video_quality_checks.insert().values(
                storyboard_id=sb_id, status="passed", dimensions=json.dumps(dims),
                issues=json.dumps(issues), created_at=stamp, updated_at=stamp))

        # ── 场景 S2：跨场景 + 无场景 ──
        e2 = make_episode(conn, did, stamp)
        seed(conn, 11, e2, stamp, image=img["a"], scene_id=1)
        seed(conn, 12, e2, stamp, image=img["a"], scene_id=2)
        seed(conn, 13, e2, stamp, image=img["down"], scene_id=None)

        # ── 场景 S3：跳过规则（无图 / 坏图 / 路径不存在） + 单镜 ──
        e3 = make_episode(conn, did, stamp)
        seed(conn, 21, e3, stamp, image=None)
        seed(conn, 22, e3, stamp, image=img["corrupt"])
        seed(conn, 23, e3, stamp, image="images/nope.png")
        e4 = make_episode(conn, did, stamp)
        seed(conn, 31, e4, stamp, image=img["a"])

        # ── 场景 S4：乱序编号 + 软删分镜（原 TS 不过滤软删 ⇒ 必须参与） ──
        # 插入序 = 3, 1, 2（故意与编号相反），并把三个 id 记下来（别靠猜）
        e5 = make_episode(conn, did, stamp)
        sb_no3 = seed(conn, 3, e5, stamp, image=img["a"], scene_id=7)
        sb_no1 = seed(conn, 1, e5, stamp, image=img["down"], scene_id=7)
        sb_no2 = seed(conn, 2, e5, stamp, image=img["up"], scene_id=7, deleted=True)

        # ── 场景 S5：端点（含软删剧集） ──
        e6 = make_episode(conn, did, stamp)
        seed(conn, 1, e6, stamp, image=img["a"])
        e7 = make_episode(conn, did, stamp, deleted=True)

    def run(eid: int):
        with engine.begin() as conn:
            return asyncio.run(run_episode_consistency_qc(conn, eid, did))

    # ── S1：分支与文案 ──
    rep1 = run(e1)
    check("报告: 只有 4 个可比对（1 号无图不参与；2 号 first_frame_image 被选中 ⇒ 共 4 对）",
          rep1["checkedPairs"] == 4 and len(rep1["pairs"]) == 4, rep1["checkedPairs"])
    check("报告: camelCase 形状 + dramaId 透传（+ ⭐ 2026-09-20 新增 consistent / coverage）",
          set(rep1) == {"episodeId", "dramaId", "checkedPairs", "warningCount", "pairs",
                        "consistent", "coverage"}
          and rep1["episodeId"] == e1 and rep1["dramaId"] == did, list(rep1))
    # ⭐⭐ 三态：**"没判 / 比过没问题 / 有问题"必须能区分** ✗（初版只有 checkedPairs+warningCount
    #    两个数 ⇒ 「一对都没比出来」与「比过且零 warning」**长得一模一样** ✗✗）
    check("⭐⭐ 三态: 真比过、但**有 warning** ⇒ consistent=False（不是 True 也不是 None）",
          rep1["consistent"] is False and rep1["warningCount"] == 2, rep1["consistent"])
    # ⭐ 自洽不变量（不写死夹具形状 ⇒ 夹具改了也不会假红 ✓）：
    #    pairs = shots − 1 ✓，且**每一对**要么"比过"要么被计数跳过 ✓（不留"消失的对" ✗）
    check("⭐ 覆盖率自洽: pairs = shots − 1，且 compared + 各类 skipped = pairs",
          rep1["coverage"]["pairs"] == rep1["coverage"]["shots"] - 1
          and (rep1["coverage"]["compared"] + rep1["coverage"]["skippedMissingImage"]
               + rep1["coverage"]["skippedUnreadable"]) == rep1["coverage"]["pairs"]
          and rep1["coverage"]["compared"] == rep1["checkedPairs"], rep1["coverage"])
    check("报告: warningCount 只数 warning 对（=2：0.422 与 0.0 两对）",
          rep1["warningCount"] == 2 and
          sum(1 for p in rep1["pairs"] if p["severity"] == "warning") == 2, rep1["warningCount"])

    p_ok, p_info, p_warn, p_zero = rep1["pairs"]
    check("同场景 sim=1.0 -> ok，且文案里整数**不带 .0**（`相似度 1，正常`）",
          p_ok["similarity"] == 1.0 and p_ok["severity"] == "ok"
          and p_ok["message"] == "同场景相邻镜头画面相似度 1，正常", p_ok)
    check("同场景 sim=0.5625 -> info，且**四舍五入到 0.563**（half-up，`round()` 会给 0.562）",
          p_info["similarity"] == 0.563 and p_info["severity"] == "info"
          and p_info["message"] == "同场景相邻镜头画面相似度 0.563 偏低，建议人工复核", p_info)
    check("同场景 sim=0.422 -> warning，文案含场景号与阈值（`同场景（#7）…≥0.55）`）",
          p_warn["severity"] == "warning" and p_warn["similarity"] == 0.422
          and p_warn["message"] == ("同场景（#7）相邻镜头画面相似度 0.422 过低（标准 ≥0.55），"
                                    "可能场景/光照/机位穿帮"), p_warn)
    check("同场景 sim=0.0 -> warning，文案里 0 渲染成 `0`（不是 0.0）",
          p_zero["similarity"] == 0.0 and p_zero["severity"] == "warning"
          and "相似度 0 过低" in p_zero["message"], p_zero["message"])
    check("配对字段: prev/cur id + 两侧图片相对路径 + sameScene 均为原始值",
          p_ok["prevStoryboardId"] == s1[0] and p_ok["curStoryboardId"] == s1[1]
          and p_ok["prevImage"] == img["a"] and p_ok["sameScene"] is True, p_ok)
    check("图片挑选: composed_image 优先，无则 first_frame_image（2 号用的就是首帧）",
          p_ok["curImage"] == img["a"] and p_info["prevImage"] == img["a"], p_info)

    # ── S1：写回 ──
    # ⚠️ 用**列表**：同一个分镜有多条 QC 记录（要断言「只更新 id 最大那条」）
    with engine.begin() as conn:
        rows = [r for r in conn.execute(
            video_quality_checks.select().order_by(video_quality_checks.c.id)).all()
            if r.storyboard_id in s1]
        newest = [r for r in rows if r.storyboard_id == s1[1] and r.id != old_qc_id][0]
        dims = json.loads(newest.dimensions)
        issues = json.loads(newest.issues)
    check("写回: 并入 dimensions.continuity_vision（保留原有键 old=1）",
          dims.get("old") == 1 and dims["continuity_vision"]["similarity"] == 1.0
          and dims["continuity_vision"]["checked"] is True
          and dims["continuity_vision"]["sameScene"] is True
          and dims["continuity_vision"]["prevStoryboardId"] == s1[0]
          and dims["continuity_vision"]["severity"] == "ok"
          and dims["continuity_vision"]["notes"] == [p_ok["message"]], dims)
    check("写回: issues 追加 `{dimension, severity, message}`（保留原有一条 x）",
          len(issues) == 2 and issues[0] == {"dimension": "x", "message": "y"}
          and issues[1] == {"dimension": "continuity_vision", "severity": "ok",
                            "message": p_ok["message"]}, issues)
    check("写回: 取 **id 最大**的 QC 记录（更旧那条 dimensions 未被改动）",
          json.loads(next(r for r in rows if r.id == old_qc_id).dimensions) == {"older": True})
    check("写回: 没有 QC 记录的分镜**不新建**记录（1 号仍无 QC）",
          not any(r.storyboard_id == s1[0] for r in rows),
          sorted((r.storyboard_id, r.id) for r in rows))
    check("写回: report 模式**不动** status（2 号仍是 passed，不是被改成别的）",
          newest.status == "passed", newest.status)

    rep1b = run(e1)
    with engine.begin() as conn:
        again = json.loads(conn.execute(
            video_quality_checks.select().where(video_quality_checks.c.id == newest.id)
        ).first().issues)
    check("重跑: 同一条 message **不重复追加** issue（去重生效 ⇒ 仍是 2 条）",
          rep1b["checkedPairs"] == 4 and len(again) == 2, again)

    # ── S2：跨场景 ──
    rep2 = run(e2)
    c1, c2 = rep2["pairs"]
    check("跨场景: 相似度 1.0（>0.9）-> info + `可能场景图复用/未切换`（**不升级 warning**）",
          c1["sameScene"] is False and c1["severity"] == "info" and c1["similarity"] == 1.0
          and c1["message"] == "跨场景切换但画面相似度 1 偏高，可能场景图复用/未切换", c1)
    check("跨场景: 低相似度 -> info + `（仅记录）`",
          c2["severity"] == "info" and c2["sameScene"] is False
          and c2["message"].endswith("（仅记录）"), c2)
    check("跨场景: **scene_id 为空**也算跨场景（`!!prev.sceneId` 语义）",
          c2["sameScene"] is False and rep2["warningCount"] == 0, rep2["warningCount"])
    check("⭐ 三态: 比过且**零 warning** ⇒ consistent=True（三态齐全：False/True/None 各有用例 ✓）",
          rep2["consistent"] is True and rep2["coverage"]["compared"] == 2, rep2["coverage"])

    # ── S3：跳过规则 —— ⭐ 2026-09-20：**「没比出来」必须能看出来** ✗ ──
    rep3 = run(e3)
    check("跳过: 无图 / 坏图（ffmpeg 解不开）/ 路径不存在 -> 全部不计入 checkedPairs",
          rep3["checkedPairs"] == 0 and rep3["pairs"] == [])
    check("⭐⭐ 一对都没比过 ⇒ consistent=None（**不拿「没判」冒充「通过」** ✗）+ 说清跳过了什么",
          rep3["consistent"] is None and rep3["coverage"]["compared"] == 0
          and (rep3["coverage"]["skippedMissingImage"] + rep3["coverage"]["skippedUnreadable"]) > 0
          and "没判" in rep3["coverage"]["note"], rep3["coverage"])
    rep4 = run(e4)
    check("单镜: 少于 2 个分镜 -> 空报告 + consistent=None + 明说「没有可比的一对」",
          rep4["checkedPairs"] == 0 and rep4["pairs"] == [] and rep4["consistent"] is None
          and rep4["coverage"]["pairs"] == 0 and "没判" in rep4["coverage"]["note"],
          rep4["coverage"])

    # ── S4：排序 + 软删不过滤 ──
    rep5 = run(e5)
    check("排序: 按 storyboard_number 排序（插入序 3,1,2 ⇒ 配对是 1→2、2→3，与插入序无关）",
          [(p["prevStoryboardId"], p["curStoryboardId"]) for p in rep5["pairs"]]
          == [(sb_no1, sb_no2), (sb_no2, sb_no3)], rep5["pairs"])
    check("软删: 原 TS **不过滤软删** ⇒ 软删的分镜照样参与比对",
          rep5["checkedPairs"] == 2, rep5["checkedPairs"])

    # ── S1：hash 缓存 ──
    calls_before = _CALLS["hash"]
    run(e1)
    # ⚠️ 是 **5** 不是 4：`asyncio.gather(prev, cur)`（TS 的 `Promise.all`）**并发**取 hash，
    # 两个协程都先看到「缓存未命中」⇒ 同一张图在**同一对**里仍会被算两次。
    # 这是**原 TS 就有的竞态**（它也是 `Promise.all` + await 后才写缓存），照抄不修；
    # 缓存真正省的是**后续对**对同一路径的重复计算。
    check("缓存: 跨对命中（并发取会重复一次 ⇒ 4 张唯一图共 5 次调用，与 Node 同竞态）",
          _CALLS["hash"] - calls_before == 5, _CALLS["hash"] - calls_before)

    # ── 常量镜像 ──
    check("阈值: 四个阈值与 TS 同名常量逐值一致（0.55/0.65/0.9/0.9）",
          consistency_qc.CONSISTENCY_QC_THRESHOLDS == {
              "sameSceneWarn": 0.55, "sameSceneInfo": 0.65,
              "sameSceneOk": 0.9, "crossSceneInfo": 0.9},
          consistency_qc.CONSISTENCY_QC_THRESHOLDS)

    # ── S5：端点 ──
    client = TestClient(app)
    ok = client.post(f"/api/v1/episodes/{e6}/consistency-qc")
    check("端点: 200 + report（单镜 ⇒ 空 pairs）", ok.status_code == 200
          and ok.json()["data"]["episodeId"] == e6
          and ok.json()["data"]["checkedPairs"] == 0, ok.text[:160])
    check("端点: 非法 id -> 404 'Invalid episode id'",
          client.post("/api/v1/episodes/abc/consistency-qc").json()["message"]
          == "Invalid episode id")
    check("端点: 剧集不存在 -> 404 'Episode not found'",
          client.post("/api/v1/episodes/999999/consistency-qc").json()["message"]
          == "Episode not found")
    check("端点: **软删剧集** -> 404 'Episode not found'",
          client.post(f"/api/v1/episodes/{e7}/consistency-qc").json()["message"]
          == "Episode not found")

    consistency_qc._compute_dhash = inner  # type: ignore[assignment]

    failed = [item for item in _R if not item[1]]
    for name, passed, detail in _R:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_R) - len(failed)}/{len(_R)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
