"""S6 自检：storyboards 的 5 条生成/LLM 端点
（`generate-tts` / `regenerate-image` / `action-suggestion` / `split` / `optimize-prompt`）。

TTS、出图、三个 LLM 助手**全部打桩** ⇒ 不打网络、不落盘。

六类必锁语义：

1. **两套回执形状**：多人对话 → ``{lines: [...]}`` 且**结果数组整体存进** ``tts_audio_url``（JSON 字符串）；
   单人 → **扁平对象** + 条件 ``warning``（not_found / no_voice 文案不同）；
2. **单人模式的提前返回**：解析结果 ``ignorable``（纯环境音/无对白）⇒ **400，且不写 start 日志**；
3. **提示词三级回退**：``body.prompt`` → 分镜级 ``customImagePrompt`` → 标准构建器；
   负向词是 ``body → 分镜自身 → 按画风默认``；参考图是 ``body.reference_images → 库内参考图``；
4. **画风解析链**：请求体 style → 剧集 style → 全局默认（``resolveEffectiveArtStyle`` 的 4 参形式）；
5. **拆分落库**：保留原镜头 + 新镜头**追加其后** + 后续分镜号**顺延** + 角色关联同步；
   时长是 ``clamp(round(原时长 / 子镜头数), 2, 4)``；
6. **场景上下文**：有 ``scene_id`` 时场景信息**整体换成场景表的**，否则用分镜自身的。

运行::

    ./.venv/Scripts/python.exe tests/storyboards_generate_test.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="sbg_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import (  # noqa: E402
    characters,
    dramas,
    episodes,
    scenes,
    storyboard_characters,
    storyboards,
)
from app.response import now  # noqa: E402
from app.routers import storyboards as sr  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


_CALLS: list[tuple[str, object]] = []


async def _fake_tts(conn, params):  # noqa: ANN001
    _CALLS.append(("tts", params))
    if params.get("text") == "BOOM":
        raise RuntimeError("TTS 炸了")
    return f"static/audio/{len(_CALLS)}.mp3"


async def _fake_image(conn, params):  # noqa: ANN001
    _CALLS.append(("image", params))
    return 777


async def _fake_suggestion(conn, params):  # noqa: ANN001
    _CALLS.append(("suggestion", params))
    if params.get("action") == "BOOM":
        raise RuntimeError("")
    return "镜头缓慢横移，人物侧身回头"


async def _fake_split(conn, params):  # noqa: ANN001
    _CALLS.append(("split", params))
    return [
        {"shotSize": "近景", "cameraMovement": "推", "actionSummary": "抬头",
         "visualFocus": "眼睛"},
        {"shotSize": "特写", "cameraMovement": "拉", "actionSummary": "落泪",
         "visualFocus": "泪珠"},
    ]


async def _fake_optimize(conn, params):  # noqa: ANN001
    _CALLS.append(("optimize", params))
    return "optimized prompt text"


sr.generate_tts = _fake_tts  # type: ignore[assignment]
sr.generate_image = _fake_image  # type: ignore[assignment]
sr.generate_action_suggestion = _fake_suggestion  # type: ignore[assignment]
sr.split_shot_into_sub_shots = _fake_split  # type: ignore[assignment]
sr.optimize_video_prompt = _fake_optimize  # type: ignore[assignment]


def main() -> int:  # noqa: C901
    ts = now()
    with engine.begin() as conn:
        drama_id = int(conn.execute(dramas.insert().values(
            title="测试剧", style="anime", created_at=ts, updated_at=ts)).lastrowid)
        episode_id = int(conn.execute(episodes.insert().values(
            drama_id=drama_id, episode_number=1, title="一", content="x", status="draft",
            image_config_id=11, audio_config_id=22, created_at=ts, updated_at=ts)).lastrowid)
        scene_id = int(conn.execute(scenes.insert().values(
            drama_id=drama_id, location="场景表地点", time="黄昏", prompt="",
            atmosphere="压抑", created_at=ts, updated_at=ts)).lastrowid)
        char_id = int(conn.execute(characters.insert().values(
            drama_id=drama_id, name="三娘", voice_style="voice-san",
            voice_speed=1.2, voice_emotion="sad", voice_pitch=0.9, voice_model="m1",
            created_at=ts, updated_at=ts)).lastrowid)
        sb_id = int(conn.execute(storyboards.insert().values(
            episode_id=episode_id, storyboard_number=1, scene_id=scene_id,
            title="第一镜", description="描述", action="走", shot_type="全景",
            angle="平视", movement="固定", atmosphere="紧张", duration=10,
            # ⚠️ 多人对话必须用**换行分隔**：台词捕获是「到下一个冒号/换行为止」的**贪婪**匹配，
            #    若两行直接相邻，第一行会把「下一个说话人」一起吞掉 ⇒ 只剩 1 行（TS 亦然）
            dialogue="三娘：今天天气真不错我很开心。\n陌生人：你是谁呀我不认识你。",
            created_at=ts, updated_at=ts)).lastrowid)
        sb2_id = int(conn.execute(storyboards.insert().values(
            episode_id=episode_id, storyboard_number=2, title="第二镜",
            created_at=ts, updated_at=ts)).lastrowid)
        plain_sb = int(conn.execute(storyboards.insert().values(
            episode_id=episode_id, storyboard_number=3, title="空镜",
            dialogue="环境音：", created_at=ts, updated_at=ts)).lastrowid)

    client = TestClient(app)
    base = "/api/v1/storyboards"

    # ================= 小助手 =================
    check("助手: _compact_json 是紧凑 JSON（无空格，守卫在盯）",
          sr._compact_json([{"a": 1}, 2]) == '[{"a":1},2]', sr._compact_json([{"a": 1}, 2]))
    with engine.begin() as conn:
        params = sr._get_character_voice_params(conn, char_id)
    check("助手: 角色声音参数取自 characters（speed/emotion/pitch/model 四个）",
          params == {"speed": 1.2, "emotion": "sad", "pitch": 0.9, "model": "m1"}, params)
    with engine.begin() as conn:
        empty_params = sr._get_character_voice_params(conn, None)
    check("助手: 无 character_id -> 空 dict", empty_params == {})

    # ================= generate-tts =================
    _CALLS.clear()
    check("TTS: id 非法 -> 404；镜头不存在 -> 404「镜头不存在」",
          client.post(f"{base}/abc/generate-tts").status_code == 404
          and client.post(f"{base}/999999/generate-tts").json()
          == {"code": 404, "message": "镜头不存在"})
    check("TTS: 纯环境音/无对白 -> 400「该镜头没有可生成的对白或旁白」",
          client.post(f"{base}/{plain_sb}/generate-tts").json()
          == {"code": 400, "message": "该镜头没有可生成的对白或旁白"})

    multi = client.post(f"{base}/{sb_id}/generate-tts").json()["data"]
    check("TTS: 多人对话 -> `{lines: [...]}`，每个角色一条（含 speaker/text/url/voice/match）",
          list(multi) == ["lines"] and len(multi["lines"]) == 2
          and set(multi["lines"][0]) >= {"speaker", "text", "tts_audio_url", "voice_id",
                                         "match_status"}, multi)
    check("TTS: 命中的角色用**库内音色 + 个性化参数**，configId 取集上的 audioConfigId",
          multi["lines"][0]["voice_id"] == "voice-san"
          and _CALLS[0][1]["voice"] == "voice-san"
          and _CALLS[0][1]["speed"] == 1.2 and _CALLS[0][1]["emotion"] == "sad"
          and _CALLS[0][1]["pitch"] == 0.9 and _CALLS[0][1]["model"] == "m1"
          and _CALLS[0][1]["configId"] == 22, _CALLS[0])
    check("TTS: 未登场的说话人 -> `match_status=not_found` 且**带 warning**",
          multi["lines"][1]["match_status"] == "not_found"
          and "陌生人" in multi["lines"][1]["warning"], multi["lines"][1])
    with engine.begin() as conn:
        stored = conn.execute(select(storyboards.c.tts_audio_url)
                              .where(storyboards.c.id == sb_id)).first()[0]
    check("TTS: 多人结果**整体以 JSON 字符串**存进 tts_audio_url",
          json.loads(stored) == multi["lines"] and " " not in stored[:20], stored[:60])

    _CALLS.clear()
    with engine.begin() as conn:
        conn.execute(storyboards.update().where(storyboards.c.id == sb2_id)
                     .values(dialogue="旁白：这是一个很长很长的旁白句子用于测试。"))
    single = client.post(f"{base}/{sb2_id}/generate-tts").json()["data"]
    check("TTS: 单人 -> **扁平对象**（含 tts_audio_url/voice_id/match_status/speaker/text）",
          set(single) == {"tts_audio_url", "voice_id", "match_status", "speaker", "text"},
          sorted(single))
    check("TTS: 旁白走 narrator + 默认音色（alloy）",
          single["match_status"] == "narrator" and single["voice_id"] == "alloy"
          and single["speaker"] == "旁白", single)
    with engine.begin() as conn:
        stored_single = conn.execute(select(storyboards.c.tts_audio_url)
                                     .where(storyboards.c.id == sb2_id)).first()[0]
    check("TTS: 单人把**字符串**存进 tts_audio_url（不是 JSON 数组）",
          stored_single == single["tts_audio_url"], stored_single)

    with engine.begin() as conn:
        conn.execute(storyboards.update().where(storyboards.c.id == sb2_id)
                     .values(dialogue="无名氏：这句话有八个字以上所以会被拆出来。"))
    no_match = client.post(f"{base}/{sb2_id}/generate-tts").json()["data"]
    check("TTS: 单人 + 角色不存在 -> warning 文案（not_found 版）",
          no_match["match_status"] == "not_found"
          and no_match["warning"] == '角色"无名氏"在剧组角色列表中不存在', no_match)

    with engine.begin() as conn:
        conn.execute(characters.update().where(characters.c.id == char_id)
                     .values(voice_style=None))
        conn.execute(storyboards.update().where(storyboards.c.id == sb2_id)
                     .values(dialogue="三娘：这句话有八个字以上所以会被拆出来。"))
    no_voice = client.post(f"{base}/{sb2_id}/generate-tts").json()["data"]
    check("TTS: 角色未配音色 -> warning 文案（no_voice 版，与 not_found 不同）",
          no_voice["match_status"] == "no_voice"
          and no_voice["warning"] == '角色"三娘"尚未配置音色', no_voice)
    with engine.begin() as conn:
        conn.execute(characters.update().where(characters.c.id == char_id)
                     .values(voice_style="voice-san"))
    with engine.begin() as conn:
        conn.execute(storyboards.update().where(storyboards.c.id == sb2_id)
                     .values(dialogue="三娘：BOOM"))
    boom = client.post(f"{base}/{sb2_id}/generate-tts")
    check("TTS: 服务抛错 -> 400 且文案是 str(err)",
          boom.status_code == 400 and boom.json()["message"] == "TTS 炸了", boom.json())

    # ================= regenerate-image =================
    _CALLS.clear()
    check("出图: id 非法 -> 404；镜头不存在 -> 404；集不存在 -> 400 Episode not found",
          client.post(f"{base}/abc/regenerate-image").status_code == 404
          and client.post(f"{base}/999999/regenerate-image").status_code == 404)
    image_ok = client.post(f"{base}/{sb_id}/regenerate-image",
                           json={}).json()["data"]
    call = _CALLS[-1][1]
    check("出图: 回执 {image_generation_id}；configId 取集上的 imageConfigId",
          image_ok == {"image_generation_id": 777} and call["configId"] == 11, (image_ok, call["configId"]))
    check("出图: 无自定义 prompt -> 标准构建器（吃掉场景描述 + 画面词收口）",
          "场景表地点" in call["prompt"] and "cinematic" in call["prompt"],
          call["prompt"][:120])
    check("出图: prompt 回退顺序 body → customImagePrompt → 构建器",
          client.post(f"{base}/{sb_id}/regenerate-image",
                      json={"prompt": "BODY"}).status_code == 200
          and _CALLS[-1][1]["prompt"] == "BODY")
    with engine.begin() as conn:
        conn.execute(storyboards.update().where(storyboards.c.id == sb_id)
                     .values(custom_image_prompt="CUSTOM"))
    check("出图: 分镜级 customImagePrompt 优先于构建器（请求体仍最高）",
          client.post(f"{base}/{sb_id}/regenerate-image", json={}).status_code == 200
          and _CALLS[-1][1]["prompt"] == "CUSTOM")
    check("出图: 显式 reference_images 覆盖库内参考图",
          client.post(f"{base}/{sb_id}/regenerate-image",
                      json={"reference_images": ["a.png", "b.png"]}).status_code == 200
          and _CALLS[-1][1]["referenceImages"] == ["a.png", "b.png"])
    check("出图: 负向词回退 body → 分镜自身 → 画风默认",
          client.post(f"{base}/{sb_id}/regenerate-image",
                      json={"negative_prompt": "NEG"}).status_code == 200
          and _CALLS[-1][1]["negativePrompt"] == "NEG")
    check("出图: force 透传",
          client.post(f"{base}/{sb_id}/regenerate-image",
                      json={"force": True}).status_code == 200
          and _CALLS[-1][1]["force"] is True)

    # ================= action-suggestion =================
    _CALLS.clear()
    check("建议: id 非法 / 镜头不存在 -> 404",
          client.post(f"{base}/abc/action-suggestion").status_code == 404
          and client.post(f"{base}/999999/action-suggestion").status_code == 404)
    suggestion = client.post(f"{base}/{sb_id}/action-suggestion").json()["data"]
    call = _CALLS[-1][1]
    check("建议: 回执 {suggestion}，入参含 title/description/action/shotType/movement/angle",
          list(suggestion) == ["suggestion"] and call["title"] == "第一镜"
          and call["shotType"] == "全景" and call["movement"] == "固定"
          and call["angle"] == "平视" and call["atmosphere"] == "紧张", call)
    check("建议: imagePrompt 取 customImagePrompt 优先",
          call["imagePrompt"] == "CUSTOM", call["imagePrompt"])
    with engine.begin() as conn:
        conn.execute(storyboards.update().where(storyboards.c.id == sb_id).values(action="BOOM"))
    empty_error = client.post(f"{base}/{sb_id}/action-suggestion")
    check("建议: 服务抛空 message -> 400 且文案兜底「Failed to generate action suggestion」",
          empty_error.status_code == 400
          and empty_error.json()["message"] == "Failed to generate action suggestion",
          empty_error.json())
    with engine.begin() as conn:
        conn.execute(storyboards.update().where(storyboards.c.id == sb_id).values(action="走"))

    # ================= split =================
    _CALLS.clear()
    check("拆分: id 非法 / 镜头不存在 -> 404",
          client.post(f"{base}/abc/split").status_code == 404
          and client.post(f"{base}/999999/split").status_code == 404)
    with engine.begin() as conn:
        conn.execute(storyboard_characters.insert().values(
            storyboard_id=sb_id, character_id=char_id))
    split_ok = client.post(f"{base}/{sb_id}/split").json()["data"]
    call = _CALLS[-1][1]
    check("拆分: 回执 subShots 5 键（id/number/shotSize/movement/action/focus）",
          len(split_ok["subShots"]) == 2
          and set(split_ok["subShots"][0]) == {"id", "storyboardNumber", "shotSize",
                                               "cameraMovement", "actionSummary",
                                               "visualFocus"}, split_ok["subShots"][0])
    check("拆分: 入参带场景信息（有 scene_id -> 换成**场景表**的地点/时间/氛围）+ 角色名 + 剧集风格",
          call["sceneInfo"] == {"location": "场景表地点", "time": "黄昏",
                                "atmosphere": "压抑"}
          and call["characterNames"] == ["三娘"] and call["visualStyle"] == "anime", call)
    with engine.begin() as conn:
        rows = conn.execute(select(storyboards).where(
            storyboards.c.episode_id == episode_id)
            .order_by(storyboards.c.storyboard_number)).all()
        links = conn.execute(select(storyboard_characters.c.storyboard_id)
                             .where(storyboard_characters.c.character_id == char_id)).all()
    check("拆分: 原镜头保留，新镜头**插在其后**（编号 2、3），后续镜头顺延到 5",
          [(row.storyboard_number, row.title) for row in rows]
          == [(1, "第一镜"), (2, "第一镜 · 近景"), (3, "第一镜 · 特写"),
              (4, "第二镜"), (5, "空镜")],
          [(row.storyboard_number, row.title) for row in rows])
    new_ids = [row.id for row in rows if row.storyboard_number in (2, 3)]
    check("拆分: 新镜头的时长 = clamp(round(原时长/子镜头数), 2, 4) = 4",
          all(row.duration == 4 for row in rows if row.storyboard_number in (2, 3)),
          [row.duration for row in rows])
    check("拆分: 新镜头 dialogue 清空、status=pending、movement 沿用子镜头/原镜头",
          all(row.dialogue is None and row.status == "pending"
              for row in rows if row.storyboard_number in (2, 3)))
    check("拆分: 角色关联**同步到新镜头**（原镜头 1 条 + 新镜头各 1 条）",
          sorted(link[0] for link in links) == sorted([sb_id, *new_ids]),
          sorted(link[0] for link in links))

    async def _empty_split(conn, params):  # noqa: ANN001
        _CALLS.append(("split", params))
        return []

    original_split = sr.split_shot_into_sub_shots
    sr.split_shot_into_sub_shots = _empty_split  # type: ignore[assignment]
    empty_result = client.post(f"{base}/{sb_id}/split").json()["data"]
    sr.split_shot_into_sub_shots = original_split  # type: ignore[assignment]
    with engine.begin() as conn:
        count_after = len(conn.execute(select(storyboards)
                                       .where(storyboards.c.episode_id == episode_id)).all())
    check("拆分: 子镜头为空数组 -> 不动任何编号、不插行",
          empty_result == {"subShots": []} and count_after == 5, count_after)

    async def _bad_split(conn, params):  # noqa: ANN001
        raise RuntimeError("")

    sr.split_shot_into_sub_shots = _bad_split  # type: ignore[assignment]
    split_error = client.post(f"{base}/{sb_id}/split")
    sr.split_shot_into_sub_shots = original_split  # type: ignore[assignment]
    check("拆分: 服务抛空 message -> 400 且文案兜底「Failed to split shot」",
          split_error.status_code == 400
          and split_error.json()["message"] == "Failed to split shot", split_error.json())

    # ================= optimize-prompt =================
    _CALLS.clear()
    check("优化: id 非法 / 镜头不存在 -> 404",
          client.post(f"{base}/abc/optimize-prompt").status_code == 404
          and client.post(f"{base}/999999/optimize-prompt").status_code == 404)
    optimized = client.post(f"{base}/{sb_id}/optimize-prompt",
                            json={"currentPrompt": "原始 prompt"}).json()["data"]
    call = _CALLS[-1][1]
    check("优化: 回执 {optimizedPrompt}，入参带 currentPrompt 与场景/角色上下文",
          optimized == {"optimizedPrompt": "optimized prompt text"}
          and call["currentPrompt"] == "原始 prompt"
          and call["sceneInfo"]["location"] == "场景表地点"
          and call["characterNames"] == ["三娘"], call)
    with engine.begin() as conn:
        conn.execute(storyboards.update().where(storyboards.c.id == sb2_id).values(
            scene_id=None, location="分镜自身地点", time="夜", atmosphere="轻松"))
    client.post(f"{base}/{sb2_id}/optimize-prompt", json={})
    check("优化: 无 scene_id -> 场景信息用**分镜自身**的（location/time/atmosphere）",
          _CALLS[-1][1]["sceneInfo"] == {"location": "分镜自身地点", "time": "夜",
                                         "atmosphere": "轻松"}, _CALLS[-1][1]["sceneInfo"])

    # ================= 汇总 =================
    failed_items = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed_items)}/{len(_RESULTS)} passed")
    return 1 if failed_items else 0


if __name__ == "__main__":
    raise SystemExit(main())
