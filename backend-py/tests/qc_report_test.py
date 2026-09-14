"""S6 自检：QC 报告交付物（``qc-report.ts`` 515 行 → `qc_report.py`）
+ ``GET /export/dramas/{id}/qc-report`` 与 ``/contact-sheet`` 两条端点。

三类交付物是**给制片/审计看的**，错在细节上不会报错只会「报告不可信」，故锁死：
**配置优先 / 探测兜底的回退链**、分数夹取与**平均值四舍五入口径**、**60 分达标边界**、
时间码式的 `toFixed(1)` 文本、以及状态 emoji 的三分支。

⚠️ 探测层（ffprobe）**不打桩**：本机有没有 ffprobe 都应返回 None（不抛），
所以断言只锁「不抛 + 回退链正确」，不锁具体探测值。

运行::

    ./.venv/Scripts/python.exe tests/qc_report_test.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="qcr_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ⚠️ Windows 控制台默认 GBK：检查名里有 `⇒`/emoji 时**打印阶段**会 UnicodeEncodeError
#    （断言其实全过了）⇒ 自己把 stdout 切到 UTF-8，别依赖外部设 PYTHONIOENCODING。
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_data_root  # noqa: E402
from app.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import (  # noqa: E402
    dramas,
    episodes,
    storyboards,
    video_generations,
    video_quality_checks,
)
from app.response import now  # noqa: E402
from app.services import qc_report as qr  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _write_static(relative: str, payload: bytes) -> str:
    absolute = os.path.join(get_data_root(), relative)
    os.makedirs(os.path.dirname(absolute), exist_ok=True)
    with open(absolute, "wb") as handle:
        handle.write(payload)
    return relative


def main() -> int:  # noqa: C901
    # ================= 助手 =================
    check("助手: pct 夹到 [0,100]，None 原样（不是 0）",
          (qr.pct(None), qr.pct(-5), qr.pct(150), qr.pct(0), qr.pct(87.5))
          == (None, 0, 100, 0, 87.5))
    check("助手: parse_json_array/obj 坏 JSON 与非目标类型都安全回退",
          qr.parse_json_array('["a",2]') == ["a", 2] and qr.parse_json_array("bad") == []
          and qr.parse_json_array('{"a":1}') == [] and qr.parse_json_array(None) == []
          and qr.parse_json_obj('{"a":1}') == {"a": 1} and qr.parse_json_obj("[1]") is None
          and qr.parse_json_obj("bad") is None)
    check("助手: `toFixed` 是 **half-away-from-zero**（1024.5 -> 1025，不是 Python 银行家舍入）",
          qr._js_to_fixed(1024.5, 0) == "1025" and qr._js_to_fixed(1953.125, 0) == "1953"
          and qr._js_to_fixed(0.05, 1) == "0.1" and qr._js_to_fixed(2.25, 1) == "2.3",
          qr._js_to_fixed(1024.5, 0))
    check("助手: `_js_num` 整数不带 `.0`（JS 的 90 不是 90.0）；缺省文案可换",
          qr._js_num(90.0) == "90" and qr._js_num(87.5) == "87.5"
          and qr._js_num(None) == "—" and qr._js_num(None, "未跑") == "未跑")
    check("助手: `_iso_now` 是 JS `toISOString` 形态（毫秒 3 位 + Z）",
          len(qr._iso_now()) == 24 and qr._iso_now().endswith("Z")
          and qr._iso_now()[10] == "T" and qr._iso_now()[19] == ".", qr._iso_now())
    check("助手: escape_html 转 &<>\"，**不转单引号**（与原 TS 一致）",
          qr.escape_html('<a href="x">&\'</a>') == "&lt;a href=&quot;x&quot;&gt;&amp;'&lt;/a&gt;",
          qr.escape_html('<a href="x">&\'</a>'))
    check("助手: 缩略图占位（无图给 placeholder，有图才出 img 且惰性加载）",
          qr.img_tag_or_placeholder(None, "首帧")
          == '<div class="thumb placeholder"><span>首帧</span></div>'
          and 'loading="lazy"' in qr.img_tag_or_placeholder("static/a.png", "首帧"))

    check("探测: 对非媒体文件不抛、返回 None（本机有无 ffprobe 都一样）",
          asyncio_run(qr.ffprobe_info(os.path.join(get_data_root(), "nope.mp4"))) is None)

    # ================= 种子 =================
    ts = now()
    video = _write_static("static/videos/sb1.mp4", b"V" * 64)
    first = _write_static("static/frames/first.png", b"f")
    last = _write_static("static/frames/last.png", b"l")
    tail = _write_static("static/frames/tail.png", b"t")
    with engine.begin() as conn:
        drama_id = int(conn.execute(dramas.insert().values(
            title="报告剧", created_at=ts, updated_at=ts)).lastrowid)
        empty_drama = int(conn.execute(dramas.insert().values(
            title="无集剧", created_at=ts, updated_at=ts)).lastrowid)
        ep1 = int(conn.execute(episodes.insert().values(
            drama_id=drama_id, episode_number=1, title="第一集", content="x",
            created_at=ts, updated_at=ts)).lastrowid)
        sb1 = int(conn.execute(storyboards.insert().values(
            episode_id=ep1, storyboard_number=1, title="镜一", description="描述" * 70,
            shot_type="近景", angle="平视", movement="推", route="keyframe",
            route_reason="含关键帧要求", status="completed", asset_status="ready",
            take_count=2, take_budget=3, duration=2.5, dialogue="台词",
            video_url=video, tts_audio_url=_write_static("static/audio/a.mp3", b"a"),
            first_frame_image=first, last_frame_image=last, tail_frame_image=tail,
            created_at=ts, updated_at=ts)).lastrowid)
        sb2 = int(conn.execute(storyboards.insert().values(
            episode_id=ep1, storyboard_number=2, title="镜二", status="pending",
            route="blocked", duration=0, created_at=ts, updated_at=ts)).lastrowid)
        # 视频生成记录：**先旧后新**（验 latest 取的是 created_at 最大的那条）
        conn.execute(video_generations.insert().values(
            drama_id=drama_id, storyboard_id=sb1, status="failed", prompt="p",
            provider="old", model="old", created_at="2026-01-01", updated_at=ts))
        conn.execute(video_generations.insert().values(
            drama_id=drama_id, storyboard_id=sb1, status="completed", prompt="p",
            provider="new", model="m1", duration=2.5, fps=24, resolution="1920x1080",
            width=1920, height=1080, seed=7, created_at="2026-02-01", updated_at=ts))
        # QC 记录：同样**先旧后新**（旧的 overall=10 不能被选中）
        conn.execute(video_quality_checks.insert().values(
            storyboard_id=sb1, status="failed", overall_score=10,
            created_at="2026-01-01", updated_at=ts))
        conn.execute(video_quality_checks.insert().values(
            storyboard_id=sb1, status="passed", lip_sync_score=88.44,
            character_consistency_score=90, continuity_score=55.55, overall_score=60,
            issues=json.dumps(["唇形偏移", "色温不一致"]), dimensions=json.dumps({"a": 1}),
            created_at="2026-02-01", updated_at=ts))

    # ================= 报告构建 =================
    with engine.begin() as conn:
        try:
            asyncio_run(qr.build_qc_report(conn, empty_drama))
            check("报告: 没有集 -> 抛 No episodes found", False)
        except ValueError as err:
            check("报告: 没有集 -> 抛 `No episodes found for QC report`",
                  str(err) == "No episodes found for QC report", str(err))
        report = asyncio_run(qr.build_qc_report(conn, drama_id))

    check("报告: 头部字段（剧名/集号/集名/时间）",
          report["dramaId"] == drama_id and report["dramaTitle"] == "报告剧"
          and report["episodeId"] == ep1 and report["episodeNumber"] == 1
          and report["episodeTitle"] == "第一集" and report["generatedAt"].endswith("Z"),
          report["episodeTitle"])
    check("报告: summary 计数（总分镜 2 / 有视频 1 / 有 QC 1）",
          report["summary"]["totalShots"] == 2 and report["summary"]["shotsWithVideo"] == 1
          and report["summary"]["shotsWithQc"] == 1, report["summary"])
    check("报告: 平均分只统计**有 QC 的镜头**，保留 1 位（`Math.round(x*10)/10`）",
          report["summary"]["avgLipSync"] == 88.4
          and report["summary"]["avgCharacterConsistency"] == 90.0
          and report["summary"]["avgContinuity"] == 55.6  # 55.55*10=555.5 -> 进位 556 -> 55.6
          and report["summary"]["avgOverall"] == 60.0, report["summary"])
    check("报告: **60 分算达标**（>=60 通过、<60 未达标）",
          report["summary"]["passed"] == 1 and report["summary"]["failed"] == 0)
    check("报告: latest 取的是 **created_at 最大**的那条（旧的 overall=10 / failed 被忽略）",
          report["shots"][0]["qc"]["status"] == "passed"
          and report["shots"][1]["qc"] is None
          and report["shots"][0]["videoGen"]["provider"] == "new",
          report["shots"][0]["videoGen"])
    check("报告: QC 分数经 pct 夹取，issues/dimensions 解析成结构化",
          report["shots"][0]["qc"]["lipSyncScore"] == 88.44
          and report["shots"][0]["qc"]["issues"] == ["唇形偏移", "色温不一致"]
          and report["shots"][0]["qc"]["dimensions"] == {"a": 1})
    check("报告: 分镜字段整体透传（route/routeReason/assetStatus/takeCount/tailFrameImage）",
          report["shots"][0]["route"] == "keyframe"
          and report["shots"][0]["routeReason"] == "含关键帧要求"
          and report["shots"][0]["assetStatus"] == "ready"
          and report["shots"][0]["takeCount"] == 2
          and report["shots"][0]["tailFrameImage"] == tail)
    check("报告: timeline 游标累加（0 / 2.5），状态来自分镜",
          [(t["storyboardNumber"], t["startSec"], t["durationSec"], t["status"])
           for t in report["timeline"]]
          == [(1, 0.0, 2.5, "completed"), (2, 2.5, 0.0, "pending")],
          report["timeline"])
    check("报告: `durationSec` 回退链 —— sb.duration 为 0 且无视频 ⇒ 0（不是 None）",
          report["shots"][1]["durationSec"] == 0)
    check("报告: media 里视频生成记录优先（duration/fps/resolution/width/height），fileSize 由本地文件补",
          report["shots"][0]["media"]["durationSec"] == 2.5
          and report["shots"][0]["media"]["fps"] == 24
          and report["shots"][0]["media"]["resolution"] == "1920x1080"
          and report["shots"][0]["media"]["fileSize"] == 64,
          report["shots"][0]["media"])
    check("报告: 无视频的分镜 media 全空（除 durationSec 回退为 0）",
          report["shots"][1]["media"]["resolution"] is None
          and report["shots"][1]["media"]["fileSize"] is None)

    # ================= Markdown =================
    with engine.begin() as conn:
        markdown = asyncio_run(qr.build_qc_report_markdown(conn, drama_id))
    lines = markdown.split("\n")
    check("MD: 标题 + 剧集/生成时间/统计行",
          lines[0] == "# QC 报告：报告剧" and lines[1] == ""
          and "- 剧集：第 1 集「第一集」" in markdown
          and "- 分镜总数：2 ｜ 有视频：1 ｜ 有 QC：1" in markdown, lines[:4])
    check("MD: 平均分用 `—` 缺省、整数不带 .0（90 而不是 90.0）",
          "- 平均分：唇形 88.4 ｜ 角色一致性 90 ｜ 连续性 55.6 ｜ 总体 60" in markdown,
          [line for line in lines if line.startswith("- 平均分")])
    check("MD: 明细表行 —— 时长 `toFixed(1)`、QC 分、状态 emoji 三分支",
          "| 1 | 近景 | 平视 | 推 | keyframe | 2.5 | ✅ | 60 | completed |" in markdown
          and "| 2 | — | — | — | blocked | 0.0 | ⛔ | — | pending |" in markdown,
          [line for line in lines if line.startswith("| 1 ") or line.startswith("| 2 ")])
    check("MD: 问题清单按镜列出（含缩进子项）",
          "- **Shot #1**（镜一）" in markdown and "  - 唇形偏移" in markdown
          and "  - 色温不一致" in markdown)
    check("MD: 时间线汇总表（起始/时长都 toFixed(1)）",
          "| 1 | 0.0 | 2.5 | completed |" in markdown
          and "| 2 | 2.5 | 0.0 | pending |" in markdown)

    with engine.begin() as conn:
        no_issue_md = asyncio_run(qr.build_qc_report_markdown(conn, empty_ep_drama(conn, drama_id)))
    check("MD: 无问题时写「无记录问题。」而不是空段", "无记录问题。" in no_issue_md)

    # ================= 联系表 HTML =================
    with engine.begin() as conn:
        html = asyncio_run(qr.build_contact_sheet_html(conn, drama_id))
    check("HTML: 联系表结构（标题/元信息/统计条/网格）",
          "联系表 Contact Sheet" in html and "<title>Contact Sheet - 报告剧</title>" in html
          and 'class="grid"' in html and "分镜 2" in html
          and "QC 达标 1 / 未达标 0" in html)
    check("HTML: 首帧/尾帧缩略图（有图出 img、无图出占位）",
          f'<img src="{first}" alt="首帧"' in html
          and f'<img src="{tail}" alt="尾帧"' in html
          and '<div class="thumb placeholder"><span>首帧</span></div>' in html)
    check("HTML: 路由徽章 —— `blocked` 加 `bad` 类（颜色变红）",
          '<span class="route-tag bad">blocked</span>' in html
          and '<span class="route-tag">keyframe</span>' in html)
    check("HTML: QC 总体三态类名（达标 qc-ok / 未达标 qc-bad / 未跑 qc-na）",
          '<span class="qc-ok">总体 60</span>' in html
          and '<span class="qc-na">总体 未跑</span>' in html)
    check("HTML: 描述截断到 120 字 + 省略号；问题只取前 3 条",
          "…</div>" in html and "问题：唇形偏移；色温不一致" in html)
    check("HTML: 规格行只在有 resolution 时出现，且 fps/时长可选拼接",
          "规格：1920x1080 @24fps 2.5s" in html)
    with engine.begin() as conn:
        report_only_html = asyncio_run(qr.build_qc_report_html(conn, drama_id))
    check("HTML: 报告版是把 Markdown 转义后塞进 <pre>（表格保留纯文本）",
          "<pre>" in report_only_html and "# QC 报告：报告剧" in report_only_html
          and "<table" not in report_only_html)

    with engine.begin() as conn:
        files = asyncio_run(qr.build_qc_report_files(conn, drama_id))
    check("文件清单: 三个交付物，标签 all（无 episodeId）",
          [f["name"] for f in files]
          == ["contact-sheet_all.html", "qc-report_all.md", "qc-report_all.html"],
          [f["name"] for f in files])
    with engine.begin() as conn:
        files_ep = asyncio_run(qr.build_qc_report_files(conn, drama_id, {"episodeId": ep1}))
    check("文件清单: 给了 episodeId 则标签 ep{id}",
          [f["name"] for f in files_ep][0] == f"contact-sheet_ep{ep1}.html")
    with engine.begin() as conn:
        media_report = asyncio_run(qr.build_media_info_report(conn, drama_id))
    check("媒体信息报告: {generatedAt, files:[media...]}（逐镜一条）",
          list(media_report) == ["generatedAt", "files"] and len(media_report["files"]) == 2
          and media_report["files"][0]["resolution"] == "1920x1080")

    # ================= 端点 =================
    client = TestClient(app)
    base = "/api/v1/export"
    check("端点: 非法 id / 剧不存在 -> 404",
          client.get(f"{base}/dramas/abc/qc-report").status_code == 404
          and client.get(f"{base}/dramas/999999/qc-report").status_code == 404
          and client.get(f"{base}/dramas/999999/contact-sheet").status_code == 404)
    check("端点: 无集的剧 -> 400（把服务异常翻译成 400，不是 500）",
          client.get(f"{base}/dramas/{empty_drama}/qc-report").json()
          == {"code": 400, "message": "No episodes found for QC report"})
    default = client.get(f"{base}/dramas/{drama_id}/qc-report")
    check("端点: qc-report 默认 json —— **紧凑** + attachment 文件名",
          default.headers["content-type"].startswith("application/json")
          and default.headers["content-disposition"]
          == f'attachment; filename="drama-{drama_id}-qc-report.json"'
          and " " not in default.text[:80]
          and default.json()["summary"]["totalShots"] == 2, default.text[:60])
    md_resp = client.get(f"{base}/dramas/{drama_id}/qc-report", params={"format": "md"})
    check("端点: format=md -> text/markdown + .md 文件名（裸文本，非信封）",
          md_resp.headers["content-type"].startswith("text/markdown")
          and md_resp.headers["content-disposition"]
          == f'attachment; filename="drama-{drama_id}-qc-report.md"'
          and md_resp.text.startswith("# QC 报告："))
    html_resp = client.get(f"{base}/dramas/{drama_id}/qc-report", params={"format": "html"})
    check("端点: format=html -> text/html；未识别值**落回 JSON**",
          html_resp.headers["content-type"].startswith("text/html")
          and client.get(f"{base}/dramas/{drama_id}/qc-report",
                         params={"format": "nope"}).headers["content-type"]
          .startswith("application/json"))
    sheet = client.get(f"{base}/dramas/{drama_id}/contact-sheet")
    check("端点: contact-sheet 文件名不带 episode；带 episodeId 时文件名带",
          sheet.headers["content-disposition"]
          == f'attachment; filename="drama-{drama_id}-contact-sheet.html"'
          and client.get(f"{base}/dramas/{drama_id}/contact-sheet",
                         params={"episodeId": str(ep1)})
          .headers["content-disposition"]
          == f'attachment; filename="drama-{drama_id}-episode-{ep1}-contact-sheet.html"',
          sheet.headers.get("content-disposition"))

    failed_items = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed_items)}/{len(_RESULTS)} passed")
    return 1 if failed_items else 0


def empty_ep_drama(conn, drama_id: int) -> int:
    """建一个「有集但无分镜」的剧，用于验证「无问题」分支。"""
    stamp = now()
    new_drama = int(conn.execute(dramas.insert().values(
        title="空剧", created_at=stamp, updated_at=stamp)).lastrowid)
    conn.execute(episodes.insert().values(
        drama_id=new_drama, episode_number=1, title="空集", content="x",
        created_at=stamp, updated_at=stamp))
    return new_drama


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)


if __name__ == "__main__":
    raise SystemExit(main())
