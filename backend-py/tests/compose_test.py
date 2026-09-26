"""compose 自检：单镜 ffmpeg 合成 + 三个端点（``ffmpeg_compose.py`` / ``routers/compose.py``）。

**不装 ffmpeg、不出真视频**：把 ``_run_ffmpeg`` 换成假的（记录参数 + 落一个空文件），
这样能精确锁住**命令参数**与**状态机**这两处最容易写歪的地方：

1. 状态流转 ``compose_processing`` → ``compose_completed`` / ``compose_failed``
   （失败时 ``composed_video_url`` 必须清空）；
2. ffmpeg 参数：``-c:v libx264 -preset fast -crf 23``、无音频 ``-an``、
   有音频 ``-map 0:v -map 1:a -c:a aac -shortest``、字幕滤镜**转义**（Windows 盘符的 ``:``）；
3. SRT 只有一条且时间轴是 ``00:00:00,500 --> 00:00:{min(duration-1,59)},000``；
4. TTS 复用策略（已有文件就不重新生成）+ 说话人 → 角色音色匹配。

运行::

    ./.venv/Scripts/python.exe tests/compose_test.py
"""
from __future__ import annotations

import asyncio
import os
import struct
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="compose_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.config import get_storage_root  # noqa: E402
from app.core.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.core.models import characters, episodes, storyboards  # noqa: E402
from app.core.response import now  # noqa: E402
from app.routers import compose as compose_router  # noqa: E402
from app.services import ffmpeg_compose as fc  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


#: 假 ffmpeg：记录调用参数 + 把输出文件建出来（``_run_ffmpeg`` 的替身）
_FFMPEG_CALLS: list[list[str]] = []
_FAIL_NEXT_FFMPEG = [False]


async def _fake_run_ffmpeg(args: list[str]) -> None:
    _FFMPEG_CALLS.append(list(args))
    if _FAIL_NEXT_FFMPEG[0]:
        _FAIL_NEXT_FFMPEG[0] = False
        raise RuntimeError("ffmpeg exited with code 1: boom")
    Path(args[-1]).write_bytes(b"fake-mp4")


#: 假 TTS：记录入参 + 返回一个「相对数据根」的路径（并把文件落下来）
_TTS_CALLS: list[dict] = []


async def _fake_generate_tts(conn, params: dict) -> str:
    _TTS_CALLS.append(dict(params))
    rel = "static/audio/tts-fake.mp3"
    target = Path(get_storage_root()) / "audio" / "tts-fake.mp3"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"fake-mp3")
    return rel


def _storyboard_row(sb_id: int):
    with engine.begin() as conn:
        return conn.execute(select(storyboards).where(storyboards.c.id == sb_id)).first()


def _compose(sb_id: int) -> str:
    return asyncio.run(fc.compose_storyboard(sb_id))


async def _noop_batch(episode_id: int, storyboard_ids: list[int]) -> None:
    """批量合成的后台循环替身。

    ⚠️ **必须隔离**：真身会开自己的事务逐镜合成，而本测试同时在用连接，
    SQLite 是**单写者** ⇒ 会一路 ``database is locked`` 重试到超时。
    路由的即时契约（标记 compose_processing + 火忘返回）不受影响。
    """


def main() -> int:  # noqa: C901
    compose_router._run_batch = _noop_batch  # type: ignore[assignment]
    fc._run_ffmpeg = _fake_run_ffmpeg  # type: ignore[assignment]
    fc.generate_tts = _fake_generate_tts  # type: ignore[assignment]
    fc.supports_subtitle_filter = lambda: True  # type: ignore[assignment]

    client = TestClient(app)
    drama_id = client.post("/api/v1/dramas", json={"title": "SMOKE compose"}).json()["data"]["id"]
    episode_id = client.get(f"/api/v1/dramas/{drama_id}").json()["data"]["episodes"][0]["id"]

    char_id = _insert_character(drama_id, "林昭", voice_style="male-qn-qingse")

    def new_sb(number: int, **fields) -> int:
        sid = client.post("/api/v1/storyboards", json={
            "episode_id": episode_id, "title": f"镜头{number}", "storyboard_number": number,
        }).json()["data"]["id"]
        if fields:
            with engine.begin() as conn:
                conn.execute(storyboards.update().where(storyboards.c.id == sid).values(**fields))
        return sid

    # ================= 正常路径：视频 + 对白 + 字幕 =================
    sb1 = new_sb(1, video_url="static/videos/v1.mp4", dialogue="林昭：你好，世界。", duration=10)
    out1 = _compose(sb1)
    row1 = _storyboard_row(sb1)
    args1 = _FFMPEG_CALLS[-1]

    check("合成: 返回 static/composed/<uuid>.mp4 且文件已落盘",
          out1.startswith("static/composed/") and out1.endswith(".mp4")
          and (Path(get_storage_root()) / "composed" / Path(out1).name).exists(),
          out1)
    check("合成: 状态 compose_completed + composed_video_url 回写",
          row1.status == "compose_completed" and row1.composed_video_url == out1,
          (row1.status, row1.composed_video_url))
    check("合成: subtitle_url 回写为 static/subtitles/<uuid>.srt",
          (row1.subtitle_url or "").startswith("static/subtitles/")
          and (row1.subtitle_url or "").endswith(".srt"),
          row1.subtitle_url)
    check("合成: tts_audio_url 回写（生成时用的是假 TTS 的返回路径）",
          row1.tts_audio_url == "static/audio/tts-fake.mp3", row1.tts_audio_url)

    srt_path = Path(get_storage_root()).parent / row1.subtitle_url
    check(
        "字幕: SRT 内容（**只有一条** + 起始 500ms + 结束 min(duration-1,59)）",
        srt_path.read_text(encoding="utf-8")
        == "1\n00:00:00,500 --> 00:00:09,000\n你好，世界。\n",
        srt_path.read_text(encoding="utf-8"),
    )
    check("合参: 输入是视频绝对路径（第一条 -i）",
          args1[0] == "ffmpeg" and args1[1] == "-i" and args1[2].endswith("v1.mp4"),
          args1[:3])
    check("合参: 音频作为第二个输入（-i 两次）", args1.count("-i") == 2, args1[:8])
    check(
        "合参: 编码参数 `-c:v libx264 -preset fast -crf 23`",
        "-c:v" in args1 and args1[args1.index("-c:v") + 1] == "libx264"
        and args1[args1.index("-preset") + 1] == "fast"
        and args1[args1.index("-crf") + 1] == "23",
        args1,
    )
    check(
        "合参: 有音频 -> `-map 0:v -map 1:a -c:a aac -shortest`（**不**加 -an）",
        "-map" in args1 and "-an" not in args1
        and args1[args1.index("-c:a") + 1] == "aac" and "-shortest" in args1,
        args1,
    )
    sub_filter = args1[args1.index("-vf") + 1] if "-vf" in args1 else ""
    check(
        "合参: 字幕滤镜 force_style 与**转义**（`\\,` 分隔 + 盘符冒号 `\\:`）",
        sub_filter.startswith("subtitles=filename='")
        and "fontsize=20" in sub_filter.lower()
        and "\\,primarycolour=&hffffff&".lower() in sub_filter.lower()
        and "\\:" in sub_filter,
        sub_filter[:150],
    )

    # ================= TTS 复用（已有文件就不再生成） =================
    existing_rel = "static/audio/existing.mp3"
    existing_abs = Path(get_storage_root()) / "audio" / "existing.mp3"
    existing_abs.parent.mkdir(parents=True, exist_ok=True)
    existing_abs.write_bytes(b"mp3")
    _TTS_CALLS.clear()
    sb2 = new_sb(2, video_url="static/videos/v2.mp4", dialogue="林昭：复用我。",
                 tts_audio_url=existing_rel, duration=6)
    _compose(sb2)
    row2 = _storyboard_row(sb2)
    check("TTS: 已有 ttsAudioUrl 且文件存在 -> **不**重新生成",
          _TTS_CALLS == [] and row2.tts_audio_url == existing_rel,
          (_TTS_CALLS, row2.tts_audio_url))
    check("TTS: 复用时音频输入是那个已有文件的绝对路径",
          _FFMPEG_CALLS[-1][4].replace("\\", "/").endswith("static/audio/existing.mp3".replace("static/", "")),
          _FFMPEG_CALLS[-1][4])
    check(
        "字幕: duration=6 -> 结束时间 00:00:05,000",
        "00:00:00,500 --> 00:00:05,000" in (
            Path(get_storage_root()).parent / row2.subtitle_url
        ).read_text(encoding="utf-8"),
    )

    # ================= TTS 说话人 → 角色音色 =================
    _TTS_CALLS.clear()
    sb3 = new_sb(3, video_url="static/videos/v3.mp4", dialogue="林昭：我的音色。", duration=8)
    _compose(sb3)
    check("TTS: 说话人按角色名匹配 -> voice 用角色音色（不是 alloy）",
          _TTS_CALLS and _TTS_CALLS[-1]["voice"] == "male-qn-qingse", _TTS_CALLS)
    check("TTS: 传入的 text 是**剥离说话人**后的正文",
          _TTS_CALLS and _TTS_CALLS[-1]["text"] == "我的音色。", _TTS_CALLS)

    _TTS_CALLS.clear()
    sb4 = new_sb(4, video_url="static/videos/v4.mp4", dialogue="路人甲：我不在剧组。", duration=8)
    _compose(sb4)
    check("TTS: 角色名不在剧组 -> 回落 alloy",
          _TTS_CALLS and _TTS_CALLS[-1]["voice"] == "alloy", _TTS_CALLS)

    # ================= 可忽略台词：不出音频、不出字幕 =================
    _FFMPEG_CALLS.clear()
    _TTS_CALLS.clear()
    sb5 = new_sb(5, video_url="static/videos/v5.mp4", dialogue="环境音", duration=7)
    _compose(sb5)
    row5 = _storyboard_row(sb5)
    check(
        "可忽略: 台词是「环境音」-> 不生成 TTS、不写字幕，但**仍然合成**（-an）",
        _TTS_CALLS == [] and row5.subtitle_url is None
        and row5.status == "compose_completed" and "-an" in _FFMPEG_CALLS[-1],
        (row5.subtitle_url, _FFMPEG_CALLS[-1]),
    )
    check("可忽略: 无音频时**没有** -map 也没有第二个 -i",
          _FFMPEG_CALLS[-1].count("-i") == 1 and "-map" not in _FFMPEG_CALLS[-1],
          _FFMPEG_CALLS[-1])

    # ================= 字幕滤镜不可用 =================
    fc.supports_subtitle_filter = lambda: False  # type: ignore[assignment]
    _FFMPEG_CALLS.clear()
    sb6 = new_sb(6, video_url="static/videos/v6.mp4", dialogue="林昭：无滤镜。", duration=5)
    _compose(sb6)
    row6 = _storyboard_row(sb6)
    check("滤镜不可用: **不烧字幕但照常合成**（SRT 仍然生成并回写）",
          "-vf" not in _FFMPEG_CALLS[-1] and row6.status == "compose_completed"
          and (row6.subtitle_url or "").endswith(".srt"),
          (row6.status, row6.subtitle_url))
    fc.supports_subtitle_filter = lambda: True  # type: ignore[assignment]

    # ================= 失败路径 =================
    _FAIL_NEXT_FFMPEG[0] = True
    sb7 = new_sb(7, video_url="static/videos/v7.mp4", dialogue="林昭：会失败。", duration=5)
    raised = ""
    try:
        _compose(sb7)
    except RuntimeError as err:
        raised = str(err)
    row7 = _storyboard_row(sb7)
    check("失败: ffmpeg 报错 -> 状态 compose_failed + composed_video_url 清空",
          row7.status == "compose_failed" and row7.composed_video_url is None, row7.status)
    check("失败: 异常继续抛出（消息透出 ffmpeg 退出码）",
          raised.startswith("ffmpeg exited with code 1"), raised[:60])

    sb8 = new_sb(8, dialogue="林昭：没有视频。")
    err8 = ""
    try:
        _compose(sb8)
    except ValueError as err:
        err8 = str(err)
    check("失败: 无 videoUrl -> 抛错且**不进**合成态", "has no video" in err8, err8)

    # ================= 路由 =================
    resp = client.post(f"/api/v1/compose/storyboards/{sb1}/compose")
    body = resp.json()
    check(
        "路由: POST /compose 路径挂在 /api/v1/compose 下，返回 **snake_case** 两字段",
        resp.status_code == 200 and set(body["data"]) == {"id", "composed_video_url"}
        and body["data"]["id"] == sb1,
        body,
    )
    check("路由: 非法 storyboard id -> 404",
          client.post("/api/v1/compose/storyboards/abc/compose").status_code == 404)
    check("路由: 无视频的分镜 -> 400（把错误消息回给前端）",
          client.post(f"/api/v1/compose/storyboards/{sb8}/compose").status_code == 400)

    # compose-all：先造「无视频」与「有视频」两种情形
    # ⚠️ `POST /episodes` 是校验型接口（字段不全返回 422，无 data 信封）⇒ 直插 DB
    empty_episode = _insert_episode(drama_id, "空集")
    check("路由: compose-all 无分镜 -> 400 No storyboards found",
          client.post(f"/api/v1/compose/episodes/{empty_episode}/compose-all").status_code == 400)

    no_video_episode = _insert_episode(drama_id, "无视频集")
    client.post("/api/v1/storyboards", json={
        "episode_id": no_video_episode, "title": "没视频", "storyboard_number": 1})
    check("路由: compose-all 分镜都没视频 -> 400 No storyboards have video yet",
          client.post(f"/api/v1/compose/episodes/{no_video_episode}/compose-all").status_code == 400)

    batch = client.post(f"/api/v1/compose/episodes/{episode_id}/compose-all").json()["data"]
    check(
        "路由: compose-all **火忘**返回 {message,total}，且只统计有视频的分镜",
        batch["total"] == 7 and batch["message"].startswith("Started composing "),
        batch,
    )
    with engine.begin() as conn:
        marked = conn.execute(
            select(storyboards.c.id, storyboards.c.status)
            .where(storyboards.c.id.in_([sb1, sb2, sb8]))
        ).all()
    statuses = {row[0]: row[1] for row in marked}
    check(
        "路由: 有视频的已标记 compose_processing，**无视频的保持原状**（仍是 pending）",
        statuses[sb1] == "compose_processing" and statuses[sb8] == "pending",
        statuses,
    )

    status_body = client.get(f"/api/v1/compose/episodes/{episode_id}/compose-status").json()["data"]
    check(
        "路由: compose-status 统计键齐全（total/completed/failed/processing/idle）",
        set(status_body) == {"total", "completed", "failed", "processing", "idle", "items"},
        list(status_body),
    )
    check("路由: compose-status 只统计有视频的分镜（total=7）", status_body["total"] == 7,
          status_body["total"])
    check("路由: processing 计入刚标记的 7 个", status_body["processing"] == 7,
          (status_body["processing"], status_body["idle"]))
    items = status_body["items"]
    check(
        "路由: items 是 **snake_case** 且含 error_msg 字段（失败时才有内容）",
        items and set(items[0]) == {
            "id", "storyboard_number", "status", "composed_video_url", "error_msg"}
        and items[0]["error_msg"] == "",
        items[0],
    )

    failed_body = client.get(f"/api/v1/compose/episodes/{episode_id}/compose-status").json()["data"]
    check(
        "路由: status 为空的分镜，items 里显示 pending（不是 null）",
        all(item["status"] for item in failed_body["items"]),
        [item["status"] for item in failed_body["items"]],
    )

    _res = client.get("/api/v1/compose/episodes/abc/compose-status")
    check("路由: compose-status 非法 id -> 404", _res.status_code == 404)

    # ================= 混音（2026-09-24 接 ✓：模型声 + 配音 → 一条音轨 ✓）=================
    # ⚠️ 原有行为是 ``-map 1:a`` **顶替** ✗ ⇒ 生成视频自带的音轨（H3 是联合 AV ✓）被**直接丢掉** ✓✗。
    #    混音是**显式开关** ✓（默认 false ⇒ 行为一字不差 ✗），判据在 ``engine/audio_mix`` ✓。
    import wave as _wave

    from app.services import segment_audio as _seg

    async def _fake_extract(video_path: str, target: str) -> str:
        """假「抽音轨」✓：写一段**真 wav**（模型声 0.5 直流 ✓）—— 不必真装 ffmpeg ✓。
        ⚠️ 必须 ``async`` ✗：真身是协程 ✓（``await`` 一个 str 会当场 TypeError ✓✗）。"""
        rows = [[0.5] * 800, [0.5] * 800]
        data = bytearray()
        for index in range(800):
            for row in rows:
                data += struct.pack("<h", int(row[index] * 32767))
        with _wave.open(target, "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(32000)
            handle.writeframes(bytes(data))
        return target

    _real_extract = fc._extract_audio_wav
    fc._extract_audio_wav = _fake_extract  # type: ignore[assignment]

    # 配音那条也得是**真 wav** ✓（假 TTS 落的是 .mp3 ✗ ⇒ 混音读不了 ✓）
    voice_wav = Path(get_storage_root()) / "audio" / "voice.wav"
    voice_wav.parent.mkdir(parents=True, exist_ok=True)
    _rows = [[0.5] * 800, [0.5] * 800]
    _data = bytearray()
    for _i in range(800):
        for _row in _rows:
            _data += struct.pack("<h", int(_row[_i] * 32767))
    with _wave.open(str(voice_wav), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(32000)
        handle.writeframes(bytes(_data))

    def _inputs(args: list[str]) -> list[str]:
        return [args[index + 1] for index, item in enumerate(args[:-1]) if item == "-i"]

    sb_mix = new_sb(90, video_url="static/videos/v2.mp4", dialogue="林昭：混音用例。", duration=10)
    with engine.begin() as conn:
        conn.execute(storyboards.update().where(storyboards.c.id == sb_mix).values(
            tts_audio_url="static/audio/voice.wav"))

    out_mix = asyncio.run(fc.compose_storyboard(sb_mix, mix_model_audio=True))
    args_mix = _FFMPEG_CALLS[-1]
    mixed_input = _inputs(args_mix)[-1]
    check("㊾ ⭐⭐ 开了混音 ⇒ ffmpeg 的第二路输入是**混合后的 wav** ✓✗（不是原配音 ✓）",
          bool(out_mix) and Path(mixed_input).name.endswith("-mixed.wav"), mixed_input)
    mixed_rows, _rate = _seg.read_wav_tracks(mixed_input)
    check("㊿ ⭐⭐ 混的是**模型声×0.6 + 配音×音量** ✓（0.5→0.8 ✓ 逐样本核 ✓）",
          abs(mixed_rows[0][0] - 0.8) < 1e-4, mixed_rows[0][:3])
    check("㊿′ 长度跟**视频（模型声）** ✓ 不跟配音 ✓✗（音轨不许把视频拖长 ✓）",
          len(mixed_rows[0]) == 800, len(mixed_rows[0]))

    sb_plain = new_sb(91, video_url="static/videos/v3.mp4", dialogue="林昭：普通用例。", duration=10)
    with engine.begin() as conn:
        conn.execute(storyboards.update().where(storyboards.c.id == sb_plain).values(
            tts_audio_url="static/audio/voice.wav"))
    asyncio.run(fc.compose_storyboard(sb_plain))
    args_plain = _FFMPEG_CALLS[-1]
    check("㊿″ ⭐ 默认（不传开关）⇒ **一字不差** ✓：音轨仍是原配音 ✓ 且**没有**混音那条路 ✓✗",
          _inputs(args_plain)[-1].endswith("voice.wav"), _inputs(args_plain))

    _bad = client.post(f"/api/v1/compose/storyboards/{sb_plain}/compose",
                       json={"voiceMode": "nope"})
    check("㊿‴ 路由：``voiceMode`` 非法 ⇒ **400 带合法值** ✗（不静默按 mix 跑 ✓✗）",
          _bad.status_code == 400 and "mix" in str(_bad.json().get("message")),
          (_bad.status_code, _bad.json().get("message")))

    fc._extract_audio_wav = _real_extract  # type: ignore[assignment]

    # ================= 汇总 =================
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


def _insert_episode(drama_id: int, title: str) -> int:
    values: dict[str, object] = {"drama_id": drama_id}
    if "title" in episodes.c:
        values["title"] = title
    if "episode_number" in episodes.c:
        with engine.begin() as conn:
            existing = conn.execute(select(episodes.c.id)).all()
        values["episode_number"] = len(existing) + 90
    for column in ("created_at", "updated_at"):
        if column in episodes.c:
            values[column] = now()
    with engine.begin() as conn:
        return int(conn.execute(episodes.insert().values(**values)).lastrowid)


def _insert_character(drama_id: int, name: str, **extra) -> int:
    values: dict[str, object] = {"drama_id": drama_id, "name": name}
    for column in ("created_at", "updated_at"):
        if column in characters.c:
            values[column] = now()
    values.update(extra)
    with engine.begin() as conn:
        return int(conn.execute(characters.insert().values(**values)).lastrowid)


if __name__ == "__main__":
    # ⚠️ Windows 中文控制台是 **GBK**（py3.14 及以前不会自动 UTF-8 ✗）⇒ 不加这行，裸跑第一条 `print` 就 `UnicodeEncodeError` 崩 ✗（守口径见 `check_cli_encoding.py` ✓）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
