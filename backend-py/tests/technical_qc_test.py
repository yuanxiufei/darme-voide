"""S7 自检：技术维度 QC（``services/technical_qc.py``）+ 与规则打分的接线。

**用 ffmpeg 现场造真实视频夹具**（黑场 / 静止画面 / 低帧率 / 大响度 / 小响度 / 干净）⇒ 每个分支都有
真值。断言刻意写成**自洽不变量**（例如「分数 = 100 − 20×黑场段数」）而不是硬编码相似度式的魔数，
这样 ffmpeg 版本差异不会造成假红。

运行::

    ./.venv/Scripts/python.exe tests/technical_qc_test.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="tqc_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.core.db import engine  # noqa: E402
from app.core.models import (  # noqa: E402
    dramas,
    episodes,
    storyboards,
    video_generations,
    video_quality_checks,
)
from app.core.response import now  # noqa: E402
from app.services import technical_qc as tq  # noqa: E402
from app.services.file_storage import get_storage_root  # noqa: E402

_R: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _R.append((name, bool(condition), detail))


def _ffmpeg(dest: Path, args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-v", "error", *args, "-y", str(dest)],
                   check=True, capture_output=True)


def build_fixtures() -> dict[str, str]:
    """在 storage root 下造夹具，返回**相对路径**（与生产存库形态一致）。"""
    root = Path(get_storage_root()) / "videos"
    root.mkdir(parents=True, exist_ok=True)

    # 纯黑 1.2s（blackdetect d=0.25 必命中）
    _ffmpeg(root / "black.mp4", ["-f", "lavfi", "-i", "color=black:s=64x64:d=1.2",
                                 "-c:v", "libx264", "-pix_fmt", "yuv420p"])
    # 纯白 2.5s：不是黑场，但**完全静止** ⇒ freezedetect 命中
    _ffmpeg(root / "white.mp4", ["-f", "lavfi", "-i", "color=white:s=64x64:d=2.5",
                                 "-c:v", "libx264", "-pix_fmt", "yuv420p"])
    # 10fps（< fpsMin 15），画面有运动 ⇒ 只该命中帧率
    _ffmpeg(root / "lowfps.mp4", ["-f", "lavfi", "-i", "testsrc2=s=64x64:r=10:d=2",
                                  "-c:v", "libx264", "-pix_fmt", "yuv420p"])
    # 干净：25fps 有运动、无音频
    _ffmpeg(root / "clean.mp4", ["-f", "lavfi", "-i", "testsrc2=s=64x64:d=2",
                                 "-c:v", "libx264", "-pix_fmt", "yuv420p"])
    # 大响度：满幅正弦（真峰必 > -1 dBTP）
    _ffmpeg(root / "loud.mp4", ["-f", "lavfi", "-i", "testsrc2=s=64x64:d=2",
                                "-f", "lavfi", "-i", "sine=frequency=1000:duration=2",
                                "-af", "volume=0dB", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                "-c:a", "aac", "-shortest"])
    # 小响度：-30dB（集成响度必偏离 -14±1 LUFS）
    _ffmpeg(root / "quiet.mp4", ["-f", "lavfi", "-i", "testsrc2=s=64x64:d=2",
                                 "-f", "lavfi", "-i", "sine=frequency=1000:duration=2",
                                 "-af", "volume=-30dB", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                 "-c:a", "aac", "-shortest"])
    # 削波素材：默认 sine ≈ -16 dBFS ⇒ +20dB 必然顶到 0dBFS（真峰 > -1dBTP 必命中）
    _ffmpeg(root / "hot.mp4", ["-f", "lavfi", "-i", "testsrc2=s=64x64:d=2",
                               "-f", "lavfi", "-i", "sine=frequency=1000:duration=2",
                               "-af", "volume=20dB", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                               "-c:a", "aac", "-shortest"])
    (root / "broken.mp4").write_bytes(b"not a video")
    return {name: f"videos/{name}.mp4"
            for name in ("black", "white", "lowfps", "clean", "loud", "quiet", "hot", "broken")}


def notes_of(spec: dict) -> list[str]:
    return [issue["message"] for issue in spec["issues"]]


def main() -> int:  # noqa: C901
    video = build_fixtures()
    stamp = now()

    with engine.begin() as conn:
        drama_id = int(conn.execute(dramas.insert().values(
            title="技术QC剧", created_at=stamp, updated_at=stamp)).lastrowid)
        episode_id = int(conn.execute(episodes.insert().values(
            drama_id=drama_id, episode_number=1, title="第一集", content="x",
            created_at=stamp, updated_at=stamp)).lastrowid)

    def seed_sb(number: int, duration: float | None = None, image: str | None = None) -> int:
        with engine.begin() as conn:
            return int(conn.execute(storyboards.insert().values(
                episode_id=episode_id, storyboard_number=number, title=f"镜{number}",
                description="d", duration=duration, created_at=stamp,
                updated_at=stamp)).lastrowid)

    def seed_gen(storyboard_id: int, local_path: str, status: str = "completed") -> int:
        with engine.begin() as conn:
            return int(conn.execute(video_generations.insert().values(
                storyboard_id=storyboard_id, provider="local", model="h3", status=status,
                local_path=local_path, created_at=stamp, updated_at=stamp)).lastrowid)

    def seed_qc(storyboard_id: int, dims: dict, issues: list) -> int:
        with engine.begin() as conn:
            return int(conn.execute(video_quality_checks.insert().values(
                storyboard_id=storyboard_id, status="passed",
                dimensions=json.dumps(dims), issues=json.dumps(issues),
                overall_score=70, created_at=stamp, updated_at=stamp)).lastrowid)

    # ── 1. 纯函数：解析助手 ──
    check("解析: blackdetect 行 -> 起止与时长",
          tq.parse_black_segments("x black_start:1.5 black_end:2.0 black_duration:0.5 y")
          == [{"start": 1.5, "duration": 0.5}])
    # ⚠️ 老形态没给 freeze_end，但**时长已知** ⇒ 不算「排到片尾」（open=False）✓
    #    初版把「没有 freeze_end」也当成 open ✗ ⇒ 文案会对一段"其实结束了"的冻帧说
    #    「一直持续到片尾」✓✗ —— 这条自检当场抓到 ✓
    check("解析: freezedetect 老形态（同一行）+ 时长已知 ⇒ open=False（不谎称排到片尾）",
          tq.parse_freeze_segments("freeze_start: 0.8 freeze_duration: 1.7")
          == [{"start": 0.8, "duration": 1.7, "end": 2.5, "open": False}])
    check("解析: volumedetect 的 mean/max（负号要留住）",
          tq.parse_volume("mean_volume: -23.4 dB\nmax_volume: -0.2 dB")
          == {"mean": -23.4, "max": -0.2})
    check("解析: ebur128 老形态（值与标题同一行）仍要吃得下",
          tq.parse_ebur128("Integrated loudness: -13.2 LUFS\nTrue peak: -0.5 dBTP")
          == {"integrated": -13.2, "truePeak": -0.5})

    # ✅ 三项检测已修（2026-09-18）：以下**全部按实测 ffmpeg 9.0.1 的真实输出**写断言 ✓
    #    （旧用例锁的是「真实格式 ⇒ 解析为空」的缺陷 ✗ —— 已按信号改成正确期望 ✓）
    check("解析[真格式] freezedetect 三行（带前缀、分行）⇒ 配对成一段",
          tq.parse_freeze_segments(
              "[Parsed_freezedetect_0 @ 0x2] lavfi.freezedetect.freeze_start: 0\n"
              "[Parsed_freezedetect_0 @ 0x2] lavfi.freezedetect.freeze_duration: 2.4\n"
              "[Parsed_freezedetect_0 @ 0x2] lavfi.freezedetect.freeze_end: 2.4")
          == [{"start": 0.0, "duration": 2.4, "end": 2.4, "open": False}])
    check("解析[真格式] 两段冻帧各成一段（事件流不能把两段粘成一段）",
          tq.parse_freeze_segments(
              "lavfi.freezedetect.freeze_start: 0\nlavfi.freezedetect.freeze_duration: 2\n"
              "lavfi.freezedetect.freeze_end: 2\nlavfi.freezedetect.freeze_start: 5\n"
              "lavfi.freezedetect.freeze_duration: 1.5\nlavfi.freezedetect.freeze_end: 6.5")
          == [{"start": 0.0, "duration": 2.0, "end": 2.0, "open": False},
              {"start": 5.0, "duration": 1.5, "end": 6.5, "open": False}])
    check("解析[真格式] 只有 freeze_start（一直冻到片尾）⇒ 用总时长收尾并标 open",
          tq.parse_freeze_segments(
              "[Parsed_freezedetect_0 @ 0x2] lavfi.freezedetect.freeze_start: 0.5", 3.0)
          == [{"start": 0.5, "duration": 2.5, "end": 3.0, "open": True}])
    check("解析: 只给 freeze_start 且**拿不到总时长** ⇒ duration=None（不猜，也不谎称片尾）",
          tq.parse_freeze_segments("lavfi.freezedetect.freeze_start: 0.5")
          == [{"start": 0.5, "duration": None, "end": None, "open": False}])
    check("解析[真格式] ebur128 summary 是**跨两行**的（值在标题下一行）",
          tq.parse_ebur128("  Integrated loudness:\n    I:         -21.8 LUFS\n"
                           "  True peak:\n    Peak:      -16.1 dBFS\n")
          == {"integrated": -21.8, "truePeak": -16.1})
    # ⚠️ 这条钉的是**新陷阱**：只有逐帧进度行时取末条 ✓ —— `I:` 是滚动累计值，
    #    首行恒为 -70 LUFS（起手静音）⇒ 取首行会造出「永远 -70」的假读数 ✗✗
    check("解析[真格式] 只有进度行 ⇒ 取**最后**一条 I:（首行恒 -70 LUFS，取首行=假读数）",
          tq.parse_ebur128(
              "[Parsed_ebur128_0 @ 0x1] t: 0.09  ...  I: -70.0 LUFS  TPK: -16.5 dBFS\n"
              "[Parsed_ebur128_0 @ 0x1] t: 1.00  ...  I: -21.8 LUFS  TPK: -16.1 dBFS\n"
              "[Parsed_ebur128_0 @ 0x1] t: 2.00  ...  I: -19.4 LUFS  TPK: -15.9 dBFS")
          == {"integrated": -19.4, "truePeak": -15.9})
    check("JS 语义: toFixed 是 half-away-from-zero（1.005 -> 1.01，Python 的 f-string 会给 1.00）",
          tq._js_to_fixed(1.005, 2) == "1.01" and tq._js_to_fixed(2.5, 0) == "3")
    check("JS 语义: 整数不带 .0（阈值 1.0 -> `1`）",
          tq._js_num_str(1.0) == "1" and tq._js_num_str(0.25) == "0.25")

    # ── 2. 黑场夹具 ──
    black = tq.probe_video_technical_spec(video["black"])
    black_notes = notes_of(black)
    # ⚠️ 纯黑夹具**同时是静止画面** ⇒ 冻帧检测修好后扣分会叠加 ✓
    #    （自洽不变量必须把两类段都算上 ⇒ 只看黑场会变成"假红" ✗）
    check("黑场: 检出一段以上，且**分数 = 100 − 20×(黑场段 + 冻帧段)**（自洽不变量）",
          len(black["blackSegments"]) >= 1
          and black["score"] == max(0, 100 - 20 * (len(black["blackSegments"])
                                                    + len(black["freezeSegments"]))), black)
    check("黑场: 报 error 级 + 文案含时长/起点/阈值（>0.25s）",
          any(i["severity"] == "error" and "画面黑场" in i["message"] and "标准 >0.25s" in i["message"]
              for i in black["issues"]), black_notes)
    check("黑场: 无音轨时 hasAudio=False 且不产生响度条目",
          black["hasAudio"] is False and black["maxVolumeDb"] is None
          and not any("响度" in m or "真峰" in m for m in black_notes), black_notes)

    # ── 3. 静止画面（冻帧）—— ✅ 已修（2026-09-18）：纯静止画面**能**检出了 ──
    white = tq.probe_video_technical_spec(video["white"])
    check("冻帧: 纯静止画面被检出（error 级 + 文案含「画面冻帧」+ 分数自洽）",
          len(white["freezeSegments"]) >= 1
          and white["score"] == max(0, 100 - 20 * len(white["freezeSegments"]))
          and any(i["severity"] == "error" and "画面冻帧" in i["message"]
                  for i in white["issues"]), white)
    check("冻帧: 该段**排到片尾**（ffmpeg 只给 freeze_start）⇒ 用总时长收尾，时长≈整片",
          white["freezeSegments"][0]["open"] is True
          and white["freezeSegments"][0]["duration"] is not None
          and abs(white["freezeSegments"][0]["duration"] - white["duration"]) < 0.5,
          (white["freezeSegments"][0], white["duration"]))
    check("冻帧: 文案点明「一直持续到片尾」（否则人以为只是一小段 ✗）",
          any("一直持续到片尾" in i["message"] for i in white["issues"]), notes_of(white))
    check("冻帧: 白场不会被误判成黑场（blackdetect 正常工作）",
          white["blackSegments"] == [] and white["duration"] is not None, white["duration"])

    # ── 4. 帧率 + 时长偏差 ──
    lowfps = tq.probe_video_technical_spec(video["lowfps"])
    check("帧率: 10fps 被检出（`帧率 10fps 过低（标准 ≥15fps）`，整数不带 .0）",
          lowfps["fps"] == 10.0
          and any(m == "帧率 10fps 过低（标准 ≥15fps）" for m in notes_of(lowfps)), lowfps["fps"])
    mismatch = tq.probe_video_technical_spec(video["lowfps"], 99)
    check("时长: 与期望偏差超 1s 时报 warning（`期望 99s`，JS 数字语义）",
          any("视频实际时长" in m and "期望 99s" in m for m in notes_of(mismatch)),
          notes_of(mismatch))
    check("时长: 期望值与实际接近时**不报**（容差 1s）",
          not any("视频实际时长" in m for m in notes_of(
              tq.probe_video_technical_spec(video["lowfps"], 2))))

    # ── 5. 干净夹具 ──
    clean = tq.probe_video_technical_spec(video["clean"])
    check("干净: 100 分 + 无 issues（notes 走「技术规格检测通过」）",
          clean["score"] == 100 and clean["issues"] == []
          and clean["blackSegments"] == [] and clean["freezeSegments"] == []
          and clean["fps"] == 25.0, clean)

    # ── 6. 音频响度 —— ✅ 已修（2026-09-18）：集成响度与真峰都能读到了 ──
    loud = tq.probe_video_technical_spec(video["loud"])
    check("响度: 有音轨 ⇒ hasAudio=True + maxVolumeDb 有值（volumedetect 正常工作）",
          loud["hasAudio"] is True and loud["maxVolumeDb"] is not None, loud["maxVolumeDb"])
    check("响度: 集成响度**读到了**（不再是恒 null ⇒ 此前这条标准永远不判 ✗）",
          loud["integratedLoudness"] is not None and -70 < loud["integratedLoudness"] < 0,
          loud["integratedLoudness"])
    # ⭐ 自洽不变量（**不写死 LUFS 数值** ⇒ ffmpeg 版本差异不会造成假红 ✓）：
    #    「报没报集成响度偏离」必须与「读数离 -14LUFS 是否 > 1」**一致** ✓
    check("响度: 「报响度偏离」⇔「读数离 -14LUFS 超 1」两侧一致（自洽不变量）",
          any("集成响度" in m for m in notes_of(loud))
          == (abs(loud["integratedLoudness"] - (-14)) > 1), notes_of(loud))
    check("响度: 默认 sine 远低于 -1dBTP ⇒ **不该**误报真峰（不该报的时候不报）",
          loud["maxVolumeDb"] < -10 and not any("真峰" in m for m in notes_of(loud)),
          (loud["maxVolumeDb"], notes_of(loud)))
    # ⭐ 反向：削波素材必须**报**真峰 —— 否则「真峰判定已生效」证明不了 ✗
    #    （一个永不触发的检查，与"没检测到"长得一模一样 ✗）
    hot = tq.probe_video_technical_spec(video["hot"])
    check("响度: 削波素材（+20dB ⇒ 顶到 0dBFS）**必须报真峰**（证明这条判定真能触发）",
          hot["hasAudio"] is True and any("真峰" in m for m in notes_of(hot)),
          (hot["maxVolumeDb"], notes_of(hot)))
    check("响度: 真峰超限扣 15 分（分数自洽）",
          hot["score"] == max(0, 100 - 15 * (1 if any("真峰" in m for m in notes_of(hot)) else 0)
                              - 10 * (1 if any("集成响度" in m for m in notes_of(hot)) else 0)
                              - 20 * (len(hot["freezeSegments"]) + len(hot["blackSegments"]))),
          hot["score"])
    quiet = tq.probe_video_technical_spec(video["quiet"])
    check("响度: -30dB 素材同样 hasAudio=True，且集成响度**更低**（读数单调）",
          quiet["hasAudio"] is True and quiet["integratedLoudness"] is not None
          and quiet["integratedLoudness"] < loud["integratedLoudness"],
          (quiet["integratedLoudness"], loud["integratedLoudness"]))
    check("响度: -30dB 素材报响度偏离（离 -14 远超 1LUFS）",
          any("集成响度" in m for m in notes_of(quiet)), notes_of(quiet))

    # ── 7. 降级分支 ──
    # ⚠️ `_run` 的判失败条件（「非零退出**且** stderr 为空」）**保持原样** ✓ ——
    #    ffmpeg 滤镜正常结束也会非零退出 ✓ ⇒ 不能改成"非零即失败" ✗。
    #    但 2026-09-20 收口了它的**静默绿灯**连带后果 ✗：坏文件读不出任何规格 ⇒ 必须报出来 ✓
    #    （初版是「满分且零 issue」✓✗ —— 坏文件看起来"干净"，是最坏的一种 ✓）。
    broken = tq.probe_video_technical_spec(video["broken"])
    check("降级: 坏文件 ⇒ **不再满分**（读不出规格 ⇒ warning + 扣 15 分），并说清后果",
          broken["score"] == 85 and broken["fps"] is None
          and any("无法读取视频规格" in m for m in notes_of(broken)), broken)
    check("降级: 坏文件**不会**被误判成有段/有音轨（没读到 ≠ 通过 ✓）",
          broken["blackSegments"] == [] and broken["freezeSegments"] == []
          and broken["hasAudio"] is False, broken)
    check("降级: 路径穿越被拒 ⇒ 带上 `error` 且字段全空（原 TS 的早退分支）",
          (lambda r: r["score"] == 100 and r.get("error") and r["issues"] == [])(
              tq.probe_video_technical_spec("../outside.mp4")))

    # ── 8. run_technical_qc 落库 ──
    sb_clean = seed_sb(1, duration=2)
    gen_clean = seed_gen(sb_clean, video["clean"])
    qc_clean = seed_qc(sb_clean, {"lip_sync": {"score": 80},
                                  "character_consistency": {"score": 70},
                                  "continuity": {"score": 60}}, [])
    with engine.begin() as conn:
        tq.run_technical_qc(conn, sb_clean, gen_clean)
        row = conn.execute(video_quality_checks.select()
                           .where(video_quality_checks.c.id == qc_clean)).first()
    dims = json.loads(row.dimensions)
    issues = json.loads(row.issues)
    check("落库: 写入 video_spec 维度（checked/score/段数为**数量**/notes 走通过文案）",
          dims["video_spec"] == {
              "score": 100, "checked": True, "notes": ["技术规格检测通过"],
              "duration": dims["video_spec"]["duration"], "fps": 25.0, "hasAudio": False,
              "maxVolumeDb": None, "integratedLoudness": None,
              "blackSegments": 0, "freezeSegments": 0}, dims["video_spec"])
    check("落库: 总分 = round(规则三维均值*0.7 + 技术分*0.3)（80/70/60 + 100 ⇒ 79）",
          row.overall_score == 79 and dims["lip_sync"]["score"] == 80, row.overall_score)
    check("落库: 无 error issue ⇒ status=passed（会**覆盖**规则打分写的状态）",
          row.status == "passed" and issues == [], (row.status, issues))

    sb_black = seed_sb(2, duration=2)
    gen_black = seed_gen(sb_black, video["black"])
    qc_black = seed_qc(sb_black, {"lip_sync": {"score": 100},
                                  "character_consistency": {"score": 100},
                                  "continuity": {"score": 100}}, [])
    with engine.begin() as conn:
        tq.run_technical_qc(conn, sb_black, gen_black)
        row_b = conn.execute(video_quality_checks.select()
                             .where(video_quality_checks.c.id == qc_black)).first()
    dims_b = json.loads(row_b.dimensions)
    issues_b = json.loads(row_b.issues)
    check("落库: error 级 issue ⇒ status=failed，并追加 `{dimension: video_spec}` 条目",
          row_b.status == "failed" and issues_b[0]["dimension"] == "video_spec"
          and issues_b[0]["severity"] == "error", issues_b)
    check("落库: 技术分下降会拉低总分（黑场扣 20×段数）",
          row_b.overall_score == round(100 * 0.7 + dims_b["video_spec"]["score"] * 0.3)
          and row_b.overall_score < 100, row_b.overall_score)

    # 重跑：issue 去重 + 维度覆盖
    with engine.begin() as conn:
        tq.run_technical_qc(conn, sb_black, gen_black)
        again = json.loads(conn.execute(video_quality_checks.select()
                                        .where(video_quality_checks.c.id == qc_black))
                           .first().issues)
    check("落库: 重跑**不重复追加**同一条 issue（去重）", len(again) == len(issues_b), again)

    # 无 local_path / 非 completed / 无 QC 记录
    sb_no_file = seed_sb(3)
    gen_no_file = seed_gen(sb_no_file, "")
    qc_no_file = seed_qc(sb_no_file, {}, [])
    sb_pending = seed_sb(4)
    gen_pending = seed_gen(sb_pending, video["clean"], status="pending")
    sb_no_qc = seed_sb(5)
    gen_no_qc = seed_gen(sb_no_qc, video["clean"])
    with engine.begin() as conn:
        tq.run_technical_qc(conn, sb_no_file, gen_no_file)
        tq.run_technical_qc(conn, sb_pending, gen_pending)
        tq.run_technical_qc(conn, sb_no_qc, gen_no_qc)
        untouched = conn.execute(video_quality_checks.select()
                                 .where(video_quality_checks.c.id == qc_no_file)).first()
    check("守卫: 无 local_path / 非 completed / 无 QC 记录 三种情况都**静默跳过**（不写坏数据）",
          json.loads(untouched.dimensions) == {} and json.loads(untouched.issues) == []
          and untouched.status == "passed", untouched.dimensions)

    # ── 9. 接线：run_qc_after_video_complete 会真的跑技术维度 ──
    sb_wired = seed_sb(6, duration=2)
    gen_wired = seed_gen(sb_wired, video["black"])
    seed_qc(sb_wired, {"lip_sync": {"score": 90}, "character_consistency": {"score": 90},
                       "continuity": {"score": 90}}, [])
    from app.services.qc_scoring import run_qc_after_video_complete  # noqa: PLC0415

    with engine.begin() as conn:
        run_qc_after_video_complete(conn, sb_wired, gen_wired)
        row_w = conn.execute(video_quality_checks.select()
                             .where(video_quality_checks.c.storyboard_id == sb_wired)
                             .order_by(video_quality_checks.c.id.desc())).first()
    dims_w = json.loads(row_w.dimensions)
    check("接线: `run_qc_after_video_complete` 已真的写回 video_spec（不再是 tech-qc-skipped 存根）",
          "video_spec" in dims_w and dims_w["video_spec"]["checked"] is True, list(dims_w))
    check("接线: 技术维度的 error 让最终状态为 failed（黑场素材）",
          row_w.status == "failed", row_w.status)

    failed = [item for item in _R if not item[1]]
    for name, passed, detail in _R:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_R) - len(failed)}/{len(_R)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
