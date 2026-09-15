"""aiVoices 自检：音色列表 / 试听 / 复刻 / 按角色批量生成 / 同步（5 端点）。

四块**错了不报错、只是结果不对**的逻辑：

1. **音色 id 生成**：``ds_``/``cv_`` 前缀 + base36 时间戳 + 4 位随机 —— 生成的 id 必须能过
   它自己那道手填校验（``^[a-zA-Z][a-zA-Z0-9_-]{7,255}$`` 且不以 ``-``/``_`` 结尾）；
2. **语言推断的顺序敏感**：``cantonese`` 要在 ``chinese`` 之前判（``粤`` 同理），
   否则粤语音色会被判成中文；``不该留`` 的音色（儿童音/播音腔）要过滤掉；
3. **``/clone`` 的三道门**：没有文件 / provider 不支持 / CosyVoice 缺 ``prompt_text``；
4. **``/generate-from-characters`` 的跳过规则**：``ds_``/``cv_`` 开头的角色音色直接 skip，
   其余要「即时合成参考音频 → 克隆 → 落音色库 → 改角色音色」。

运行::

    ./.venv/Scripts/python.exe tests/ai_voices_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="aivoice_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.config import get_storage_root  # noqa: E402
from app.core.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.core.models import ai_service_configs, ai_voices, characters  # noqa: E402
from app.core.response import now  # noqa: E402
from app.routers import ai_voices as av  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


_TTS_CALLS: list[dict] = []
_CLONE_CALLS: list[dict] = []
_COSY_CALLS: list[dict] = []
_TAG_CALLS: list[list] = []
_VENDOR = [{"base_resp": {"status_code": 0}, "system_voice": []}]


async def _fake_generate_tts(conn, params: dict) -> str:
    _TTS_CALLS.append(dict(params))
    rel = "static/audio/ref.mp3"
    target = Path(get_storage_root()) / "audio" / "ref.mp3"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"mp3")
    return rel


async def _fake_clone(input_data: dict) -> dict:
    _CLONE_CALLS.append(dict(input_data))
    return {"demoAudio": "static/audio/demo.mp3"}


async def _fake_clone_cosy(input_data: dict) -> dict:
    _COSY_CALLS.append(dict(input_data))
    return {"demoAudio": "aGk="}  # base64("hi")


async def _fake_infer_tags(voices: list) -> dict:
    _TAG_CALLS.append([dict(v) for v in voices])
    return {v["voiceId"]: ["主角"] for v in voices}


async def _fake_fetch(base_url: str, api_key: str):
    return dict(_VENDOR[0]), None


def _row(table, row_id):
    with engine.begin() as conn:
        return conn.execute(select(table).where(table.c.id == row_id)).first()


def main() -> int:  # noqa: C901
    # ⚠️ 一律 patch **路由模块**里的名字（路由是 `from ... import x`）
    av.generate_tts = _fake_generate_tts  # type: ignore[assignment]
    av.clone_voice = _fake_clone  # type: ignore[assignment]
    av.clone_voice_cosyvoice = _fake_clone_cosy  # type: ignore[assignment]
    av.infer_voice_role_tags = _fake_infer_tags  # type: ignore[assignment]
    av.fetch_minimax_voices = _fake_fetch  # type: ignore[assignment]

    # ================= 纯函数 =================
    check("id: 手填合法 id 原样返回", av.generate_clone_voice_id("voice_abcd1234") == "voice_abcd1234")
    bad = ""
    try:
        av.generate_clone_voice_id("short")
    except ValueError as exc:
        bad = str(exc)
    check("id: 手填太短 -> 抛中文校验错",
          bad == "voice_id 需 8-256 字符，首字符为英文字母，仅含字母/数字/-/_", bad)
    check("id: 以 `-`/`_` 结尾的 id 也拒绝",
          _raises(lambda: av.generate_clone_voice_id("voice_abcd123_")), None)
    generated = av.generate_clone_voice_id("", "ds_")
    cosy_generated = av.generate_clone_voice_id("", "cv_")
    check(
        "id: 生成的 id 以 ds_/cv_ 开头，且**能过自己那道手填校验**（自洽）",
        generated.startswith("ds_") and cosy_generated.startswith("cv_")
        and av.generate_clone_voice_id(generated) == generated
        and len(generated) >= 8,
        generated,
    )

    check(
        "语言: 顺序敏感 —— 粤语优先于中文（cantonese / 粤）",
        av.extract_language("cantonese_1", "") == "粤语"
        and av.extract_language("voice", "粤语男声") == "粤语"
        and av.extract_language("chinese_1", "") == "中文"
        and av.extract_language("x", "中文女声") == "中文",
    )
    check("语言: 英/日/韩等也能识别", (av.extract_language("english_x", ""), av.extract_language("japanese_x", ""),
                                av.extract_language("korean_x", "")) == ("英语", "日语", "韩语"))
    check("语言: 未知 -> 其他", av.extract_language("zzz", "???") == "其他")
    check(
        "过滤: 非中文/粤语一律丢掉；中文里的排除项（播音腔等）也丢掉",
        av.should_keep_voice("english_male", "") is False
        and av.should_keep_voice("Chinese_news_anchor", "") is False
        and av.should_keep_voice("Chinese_male_qn", "") is True,
    )
    check("解析: parse_json_array 容错（脏数据/非数组 -> 空）",
          av.parse_json_array('["a",1]') == ["a", "1"] and av.parse_json_array("nope") == []
          and av.parse_json_array('{"a":1}') == [] and av.parse_json_array(None) == [])
    check("角色: map_role_to_tag（主角/反派/旁白原样，龙套与空 -> 配角）",
          [av.map_role_to_tag(r) for r in ("主角", "反派", "旁白", "龙套", None)]
          == ["主角", "反派", "旁白", "配角", "配角"])
    check(
        "路径: resolve_local_audio_path（剥 static、去 query、空 -> None）",
        av.resolve_local_audio_path("static/audio/a.mp3").replace("\\", "/").endswith("audio/a.mp3")
        and av.resolve_local_audio_path("/static/audio/a.mp3?v=1").replace("\\", "/").endswith("audio/a.mp3")
        and av.resolve_local_audio_path("") is None,
    )

    client = TestClient(app)

    # ================= GET / =================
    _insert_voice("minimax", "Chinese_male_qn", "青涩青年", '["青年音"]', role_tags='["主角"]')
    _insert_voice("minimax", "Chinese_female_yun", "云希", None, role_tags=None)
    _insert_voice("cosyvoice", "cosy_1", "克隆音色", '["克隆音色"]', role_tags="bad json")
    listed = client.get("/api/v1/ai-voices").json()["data"]
    check("列表: 默认 provider=minimax（不含 cosyvoice）",
          len(listed) == 2 and all(row["provider"] == "minimax" for row in listed), len(listed))
    check(
        "列表: 字段是 snake_case 且 description/role_tags 已解析成数组",
        listed[0]["voice_id"] == "Chinese_male_qn" and listed[0]["description"] == ["青年音"]
        and listed[0]["role_tags"] == ["主角"],
        listed[0],
    )
    check("列表: description 为空 -> []；role_tags 脏数据 -> []（容错）",
          listed[1]["description"] == [] and listed[1]["role_tags"] == [],
          (listed[1]["description"], listed[1]["role_tags"]))
    check("列表: 指定 provider=cosyvoice",
          len(client.get("/api/v1/ai-voices?provider=cosyvoice").json()["data"]) == 1)

    # ================= POST /preview =================
    check("试听: 缺 voice_id -> 400",
          client.post("/api/v1/ai-voices/preview", json={}).json()["message"] == "voice_id is required")
    preview = client.post("/api/v1/ai-voices/preview", json={"voice_id": "v1"}).json()["data"]
    check(
        "试听: 用**默认文案**合成，返回 {voice_id, url}",
        preview == {"voice_id": "v1", "url": "static/audio/ref.mp3"}
        and _TTS_CALLS[-1]["text"] == av.DEFAULT_SAMPLE_TEXT,
        preview,
    )
    _TTS_CALLS.clear()
    client.post("/api/v1/ai-voices/preview", json={"voice_id": "v1", "text": "自定义", "config_id": 7})
    check("试听: 可传自定义文案与 config_id",
          _TTS_CALLS[-1]["text"] == "自定义" and _TTS_CALLS[-1]["configId"] == 7, _TTS_CALLS[-1])

    # ================= POST /clone =================
    check("复刻: 没有文件 -> 400 file is required",
          client.post("/api/v1/ai-voices/clone", data={}).json()["message"] == "file is required")

    _set_audio_config("openai")  # 不支持的厂商
    unsupported = client.post("/api/v1/ai-voices/clone", data={},
                              files={"file": ("a.wav", b"xyz", "audio/wav")})
    check("复刻: provider 非 minimax/cosyvoice -> 400 且点明当前厂商",
          "仅支持 MiniMax / CosyVoice" in unsupported.json()["message"]
          and "openai" in unsupported.json()["message"],
          unsupported.json()["message"])

    _set_audio_config("cosyvoice")
    need_prompt = client.post("/api/v1/ai-voices/clone", data={},
                              files={"file": ("a.wav", b"xyz", "audio/wav")})
    check("复刻: cosyvoice 缺 prompt_text -> 400 专属文案",
          "需要提供参考音频文本 prompt_text" in need_prompt.json()["message"],
          need_prompt.json()["message"])

    cosy = client.post("/api/v1/ai-voices/clone",
                       data={"voice_name": "我的克隆", "prompt_text": "参考文本"},
                       files={"file": ("a.wav", b"xyz", "audio/wav")}).json()["data"]
    check(
        "复刻(cosy): 生成 cv_ 前缀 id；demo 音频落盘为 static/audio/*.mp3",
        cosy["voice_id"].startswith("cv_") and (cosy["demo_audio"] or "").startswith("static/audio/"),
        cosy,
    )
    with engine.begin() as conn:
        stored = conn.execute(select(ai_voices).where(
            ai_voices.c.voice_id == cosy["voice_id"])).first()
    check(
        "复刻(cosy): 库里存了参考音频（static/voices/*.wav）与参考文本",
        (stored.reference_audio or "").startswith("static/voices/")
        and stored.prompt_text == "参考文本" and stored.provider == "cosyvoice",
        (stored.reference_audio, stored.prompt_text),
    )
    check("复刻(cosy): 参考音频文件真的落盘了",
          Path(av.resolve_local_audio_path(stored.reference_audio)).exists(),
          stored.reference_audio)

    _set_audio_config("minimax")
    mini = client.post("/api/v1/ai-voices/clone", data={},
                       files={"file": ("a.wav", b"xyz", "audio/wav")}).json()["data"]
    with engine.begin() as conn:
        mini_row = conn.execute(select(ai_voices).where(
            ai_voices.c.voice_id == mini["voice_id"])).first()
    check(
        "复刻(minimax): id 走 ds_ 前缀，名称回落 `克隆音色 <后6位>`",
        mini["voice_id"].startswith("ds_")
        and mini_row.voice_name == f"克隆音色 {mini['voice_id'][-6:]}",
        (mini["voice_id"], mini_row.voice_name),
    )
    check("复刻: demo_audio 透传厂商返回", mini["demo_audio"] == "static/audio/demo.mp3", mini)
    # 冲突不覆盖：同 id 再插一次，名称不该被改
    with engine.begin() as conn:
        av._insert_voice_ignore_conflict(conn, {
            "voice_id": mini["voice_id"], "voice_name": "不该覆盖", "language": "中文",
            "provider": "minimax", "created_at": now(),
        })
    with engine.begin() as conn:
        again = conn.execute(select(ai_voices).where(
            ai_voices.c.voice_id == mini["voice_id"])).first()
    check("复刻: onConflictDoNothing —— 同 id **不覆盖**已有音色",
          again.voice_name != "不该覆盖", again.voice_name)

    # ================= POST /generate-from-characters =================
    drama_id = client.post("/api/v1/dramas", json={"title": "SMOKE 音色"}).json()["data"]["id"]
    check("批量: 缺 drama_id -> 400",
          client.post("/api/v1/ai-voices/generate-from-characters", json={}).json()["message"]
          == "drama_id is required")
    check("批量: 剧不存在 -> 404",
          client.post("/api/v1/ai-voices/generate-from-characters",
                      json={"drama_id": 999999}).status_code == 404)
    check("批量: 没有已分配音色的角色 -> 400 中文提示",
          "暂无已分配音色的角色" in client.post(
              "/api/v1/ai-voices/generate-from-characters",
              json={"drama_id": drama_id}).json()["message"])

    fresh = _insert_character(drama_id, "林昭", voice_style="Chinese_male_qn", role="主角")
    done_char = _insert_character(drama_id, "阿晚", voice_style="ds_already", role="龙套")
    _TTS_CALLS.clear()
    _CLONE_CALLS.clear()
    batch = client.post("/api/v1/ai-voices/generate-from-characters",
                        json={"drama_id": drama_id}).json()["data"]
    statuses = {row["name"]: row["status"] for row in batch["results"]}
    check(
        "批量: 汇总 total/success_count + 逐角色结果",
        batch["total"] == 2 and batch["success_count"] == 1
        and statuses == {"林昭": "success", "阿晚": "skipped"},
        batch,
    )
    check("批量: 已有 ds_ 前缀的角色音色 -> skipped（reason 已有专属音色）",
          [row for row in batch["results"] if row["name"] == "阿晚"][0]["reason"] == "已有专属音色")
    ref_text = _TTS_CALLS[-1]["text"]
    check(
        "批量: 参考音频用**约 15 秒长文本**（MiniMax 克隆要求 ≥10 秒，试听 4 秒会报 too short）",
        ref_text == av.REF_TEXT and len(ref_text) > 60, len(ref_text),
    )
    check("批量: 克隆用的是**即时合成的参考音频文件**（不是试听 URL）",
          _CLONE_CALLS[-1]["filename"] == "ref.mp3", _CLONE_CALLS[-1]["filename"])
    new_id = [row for row in batch["results"] if row["name"] == "林昭"][0]["voice_id"]
    char_row = _row(characters, fresh)
    check(
        "批量: 角色 voice_style 改为专属音色 + voice_provider 落库",
        char_row.voice_style == new_id and char_row.voice_provider == "minimax",
        (char_row.voice_style, char_row.voice_provider),
    )
    with engine.begin() as conn:
        voice_row = conn.execute(select(ai_voices).where(ai_voices.c.voice_id == new_id)).first()
    check(
        "批量: 音色名用角色名、description 用角色 role、role_tags 是四类之一",
        voice_row.voice_name == "林昭" and voice_row.description == '["主角"]'
        and voice_row.role_tags == '["主角"]',
        (voice_row.voice_name, voice_row.description, voice_row.role_tags),
    )
    check("批量: 跳过的那位角色音色**没被动过**",
          _row(characters, done_char).voice_style == "ds_already")

    # ================= POST /sync =================
    # ⚠️ 先换成**非 minimax** 的活跃配置，否则会命中下面的正常分支（前面的复刻用例留了 minimax）
    _set_audio_config("openai")
    check("同步: 没有活跃 minimax 音频配置 -> 400",
          client.post("/api/v1/ai-voices/sync").json()["message"]
          == "No active minimax audio config found",
          client.post("/api/v1/ai-voices/sync").json()["message"])
    _set_audio_config("minimax", api_key=None)
    check("同步: 配置缺 api_key -> 400",
          client.post("/api/v1/ai-voices/sync").json()["message"] == "MiniMax API key not configured")

    _set_audio_config("minimax", api_key="k")
    _VENDOR[0] = {"base_resp": {"status_code": 1004, "status_msg": "鉴权失败"}}
    check("同步: base_resp.status_code != 0 -> 400 透出 status_msg",
          client.post("/api/v1/ai-voices/sync").json()["message"] == "鉴权失败")

    _VENDOR[0] = {
        "base_resp": {"status_code": 0},
        "system_voice": [
            {"voice_id": "Chinese_male_qn", "voice_name": "青涩青年", "description": ["青年"]},
            {"voice_id": "Chinese_news_anchor", "voice_name": "新闻主播", "description": []},
            {"voice_id": "english_male", "voice_name": "English Male", "description": []},
        ],
    }
    _TAG_CALLS.clear()
    synced = client.post("/api/v1/ai-voices/sync").json()["data"]
    check(
        "同步: 只留中文且非排除项（英文/播音腔被过滤）",
        synced["count"] == 1 and synced["message"] == "Synced 1 voices", synced,
    )
    remaining = client.get("/api/v1/ai-voices").json()["data"]
    check(
        "同步: **先清空旧数据**再插入（旧音色不再出现在列表里）",
        [row["voice_id"] for row in remaining] == ["Chinese_male_qn"],
        [row["voice_id"] for row in remaining],
    )
    check("同步: 触发 AI 打标并回写 role_tags",
          _TAG_CALLS and remaining[0]["role_tags"] == ["主角"], remaining[0]["role_tags"])

    # ================= 汇总 =================
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:  # noqa: BLE001
        return True
    return False


def _insert_voice(provider: str, voice_id: str, name: str, description: str | None,
                  role_tags: str | None = None) -> int:
    with engine.begin() as conn:
        return int(conn.execute(ai_voices.insert().values(
            voice_id=voice_id, voice_name=name, description=description,
            role_tags=role_tags, language="中文", provider=provider, created_at=now(),
        )).lastrowid)


def _insert_character(drama_id: int, name: str, **extra) -> int:
    values: dict[str, object] = {"drama_id": drama_id, "name": name}
    for column in ("created_at", "updated_at"):
        if column in characters.c:
            values[column] = now()
    values.update(extra)
    with engine.begin() as conn:
        return int(conn.execute(characters.insert().values(**values)).lastrowid)


def _set_audio_config(provider: str, api_key: str | None = "k") -> None:
    """把 audio 服务配置换成指定厂商（同一时刻只留一条 active）。"""
    with engine.begin() as conn:
        conn.execute(ai_service_configs.update().values(is_active=0))
        values = {
            "service_type": "audio", "provider": provider, "base_url": "https://api.example.com",
            # api_key 是 NOT NULL ⇒ 用空串表达「未配置」（`!apiKey` 判定与 JS 一致）
            "api_key": api_key or "", "model": "speech-2.8-hd", "is_active": 1,
        }
        # name / 时间戳是 NOT NULL（漏了会 IntegrityError）
        for column in ("name", "created_at", "updated_at"):
            if column in ai_service_configs.c:
                values[column] = f"测试音频配置-{provider}" if column == "name" else now()
        conn.execute(ai_service_configs.insert().values(**values))


if __name__ == "__main__":
    raise SystemExit(main())
