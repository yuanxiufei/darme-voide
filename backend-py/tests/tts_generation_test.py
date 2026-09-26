"""TTS 链路自检（``app/services/tts_generation.py`` + ``voice_clone.py``）。

用 ``httpx.MockTransport`` 拦截全部厂商请求（**零真实网络**），并**捕获请求体**做保真断言
（例如 hex 解码是截断而非抛错、``file_id`` 要发成**数字**、multipart 的字段名与 MIME）。

同步链路（TTS）与异步链路（图片/视频）最大的差别在这里体现：**没有轮询、直接落盘**。

运行::

    ./.venv/Scripts/python.exe tests/tts_generation_test.py
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="ttsgen_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.config import get_storage_root  # noqa: E402
from app.core.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.core.models import ai_voices, api_usage  # noqa: E402
from app.core.response import now  # noqa: E402
from app.services import tts_generation as tts  # noqa: E402
from app.services import vendor_errors  # noqa: E402
from app.services import voice_clone as vc  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def run(coro):
    return asyncio.run(coro)


def _install(handler):
    """把全局厂商客户端换成 MockTransport（``_get_client()`` 返回它）。"""
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    vendor_errors._vendor_client = client  # type: ignore[assignment]
    return client


def main() -> int:  # noqa: C901
    client = TestClient(app)

    # ================= 纯函数 =================
    check("hex: 正常成对解码", tts._hex_to_bytes("6869") == b"hi", tts._hex_to_bytes("6869"))
    check("hex: 大写也认", tts._hex_to_bytes("6869AB") == bytes.fromhex("6869ab"))
    check(
        "hex: 遇到非法字符即**截断**（不是抛错，对齐 Node 的 Buffer.from）",
        tts._hex_to_bytes("68zz") == b"\x68" and tts._hex_to_bytes("zz") == b"",
        (tts._hex_to_bytes("68zz"), tts._hex_to_bytes("zz")),
    )
    check("hex: 奇数长度丢掉最后一个半字节", tts._hex_to_bytes("686") == b"h", tts._hex_to_bytes("686"))
    check("hex: 空串 -> 空字节", tts._hex_to_bytes("") == b"" and tts._hex_to_bytes(None) == b"")

    check(
        "mime: wav / m4a / mp4 / 默认 mp3，且大小写不敏感",
        [vc.infer_audio_mime(n) for n in ("a.wav", "a.m4a", "a.MP4", "a.mp3", "a.WAV")]
        == ["audio/wav", "audio/m4a", "audio/m4a", "audio/mpeg", "audio/wav"],
        [vc.infer_audio_mime(n) for n in ("a.wav", "a.m4a", "a.MP4", "a.mp3", "a.WAV")],
    )

    # ================= 配置 =================
    client.post("/api/v1/ai-configs", json={
        "service_type": "audio", "provider": "minimax", "base_url": "https://api.minimax.test",
        "api_key": "k", "model": ["speech-2.8-hd"], "is_active": True,
    })

    # ================= generate_tts：MiniMax（hex 音频） =================
    seen: list[httpx.Request] = []

    def minimax_handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if "t2a_v2" in str(request.url):
            return httpx.Response(200, json={
                "base_resp": {"status_code": 0},
                "data": {
                    "audio": "68656c6c6f",  # "hello"
                    "extra_info": {"audio_length": 1500, "audio_format": "mp3"},
                },
            })
        return httpx.Response(404, text="nope")

    _install(minimax_handler)
    with engine.begin() as conn:
        path = run(tts.generate_tts(conn, {"text": "你好世界", "voice": "v1", "speed": 1, "emotion": "happy"}))
    check(
        "tts: 返回相对路径 static/audio/<uuid>.mp3",
        path.startswith("static/audio/") and path.endswith(".mp3"), path,
    )
    written = Path(get_storage_root()) / "audio" / Path(path).name
    check("tts: 落盘内容是 hex 解码后的字节", written.read_bytes() == b"hello", written.read_bytes())
    check(
        "tts: 请求打到 /v1/t2a_v2、带 Bearer、用 JSON 紧凑体",
        len(seen) == 1 and str(seen[0].url).endswith("/v1/t2a_v2")
        and seen[0].headers.get("authorization") == "Bearer k"
        and b'{"model":"speech-2.8-hd"' in seen[0].content,
        str(seen[0].url),
    )
    check(
        "tts: 请求体里的 emotion 已映射为 MiniMax 值（happy）",
        json.loads(seen[0].content)["voice_setting"]["emotion"] == "happy",
        json.loads(seen[0].content)["voice_setting"],
    )
    with engine.begin() as conn:
        usage = conn.execute(select(api_usage)).all()
    check(
        "tts: 用量按**字符数**记账（units=4）且状态直接 completed（同步链路无 submitted 阶段）",
        len(usage) == 1 and usage[0].units == 4 and usage[0].status == "completed"
        and usage[0].service_type == "audio",
        (len(usage), usage[0].units if usage else None, usage[0].status if usage else None),
    )

    # ================= 多模型 fallback =================
    client.post("/api/v1/ai-configs", json={
        "service_type": "audio", "provider": "minimax", "base_url": "https://api.minimax.test",
        "api_key": "k", "model": ["bad-model", "good-model"], "is_active": True, "priority": 5,
    })
    attempts: list[str] = []

    def fallback_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        attempts.append(body.get("model"))
        if body.get("model") == "bad-model":
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json={"base_resp": {"status_code": 0}, "data": {"audio": "aabb"}})

    _install(fallback_handler)
    with engine.begin() as conn:
        before = len(conn.execute(select(api_usage)).all())
        ok_path = run(tts.generate_tts(conn, {"text": "x", "voice": "v"}))
        after = len(conn.execute(select(api_usage)).all())
    # ⚠️ 500 会被 `fetch_with_retry` 重试 **3 次**（TTS 不传 maxRetries ⇒ 用默认 3），
    #    所以第一个模型会发 3 个请求，然后才轮到第二个模型。
    #    另外注意 detail 要传**副本**：传 list 对象的话，后面的 clear()/append 会让
    #    打印出来的内容与断言当时的状态不一致（这个坑本次就踩到了）。
    check(
        "fallback: 第一个模型 500（重试 3 次）后切到第二个并成功",
        attempts == ["bad-model"] * 3 + ["good-model"] and ok_path.startswith("static/audio/"),
        list(attempts),
    )
    check(
        "fallback: **失败的那次不记账**（recordUsage 只在成功后调用）",
        after - before == 1, (before, after),
    )

    # params.model 会关掉 fallback
    attempts.clear()
    with engine.begin() as conn:
        try:
            run(tts.generate_tts(conn, {"text": "x", "voice": "v", "model": "bad-model"}))
            check("fallback: 显式传 model 时 fallback 失效（每次都用它）", False, "未抛错")
        except ValueError:
            check(
                "fallback: 显式传 model 时 fallback 失效（两个模型都发同一个名字）—— 原实现的怪癖",
                attempts == ["bad-model"] * 6, list(attempts),
            )

    # 全部失败
    def all_fail(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    _install(all_fail)
    with engine.begin() as conn:
        try:
            run(tts.generate_tts(conn, {"text": "x", "voice": "v"}))
            check("fallback: 全部模型失败 -> 抛出", False, "未抛错")
        except Exception as err:  # noqa: BLE001
            check("fallback: 全部模型失败 -> 抛出（最后一次的归因中文）",
                  "服务暂时异常" in str(err) or "请求较多" in str(err), str(err)[:60])

    # ================= CosyVoice：base64 -> hex + 零样本 =================
    storage = Path(get_storage_root())
    sample_dir = storage / "audio"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_file = sample_dir / "ref.wav"
    sample_file.write_bytes(b"RIFF-fake-wav")
    with engine.begin() as conn:
        values = {"voice_id": "cosy-1", "voice_name": "测试音色", "provider": "cosyvoice",
                  "reference_audio": "static/audio/ref.wav", "prompt_text": "参考文本"}
        if "created_at" in ai_voices.c:
            values["created_at"] = now()
        conn.execute(ai_voices.insert().values(**values))

    client.post("/api/v1/ai-configs", json={
        "service_type": "audio", "provider": "cosyvoice", "base_url": "http://127.0.0.1:9880",
        "api_key": "k", "model": ["cosyvoice-v2"], "is_active": True, "priority": 9,
    })
    cosy_requests: list[httpx.Request] = []

    def cosy_handler(request: httpx.Request) -> httpx.Response:
        cosy_requests.append(request)
        if str(request.url).endswith("/inference_zero_shot"):
            # 零样本：返回 base64 音频（适配器会转成 hex）
            return httpx.Response(200, json={"audio": base64.b64encode(b"zero-shot").decode("ascii")})
        return httpx.Response(200, json={"audio": base64.b64encode(b"normal").decode("ascii")})

    _install(cosy_handler)
    with engine.begin() as conn:
        cosy_path = run(tts.generate_tts(conn, {"text": "合成", "voice": "cosy-1"}))
    check(
        "cosy: 音色库有参考音频 -> 走 /inference_zero_shot（零样本复用）",
        str(cosy_requests[0].url).endswith("/inference_zero_shot"), str(cosy_requests[0].url),
    )
    cosy_body = json.loads(cosy_requests[0].content)
    check(
        "cosy: 请求体带 base64 的 prompt_audio + prompt_text",
        cosy_body["prompt_audio"] == base64.b64encode(b"RIFF-fake-wav").decode("ascii")
        and cosy_body["prompt_text"] == "参考文本" and cosy_body["tts_text"] == "合成",
        {k: str(v)[:24] for k, v in cosy_body.items()},
    )
    check(
        "cosy: base64 音频被适配器转成 hex 后落盘（内容正确）",
        (Path(get_storage_root()) / "audio" / Path(cosy_path).name).read_bytes() == b"zero-shot",
        (Path(get_storage_root()) / "audio" / Path(cosy_path).name).read_bytes(),
    )

    # 参考音频文件不存在 -> 静默退回普通 /tts
    with engine.begin() as conn:
        conn.execute(ai_voices.update().where(ai_voices.c.voice_id == "cosy-1").values(
            reference_audio="static/audio/missing.wav"))
    cosy_requests.clear()
    _install(cosy_handler)
    with engine.begin() as conn:
        plain_path = run(tts.generate_tts(conn, {"text": "合成", "voice": "cosy-1"}))
    check(
        "cosy: 参考音频**文件不存在** -> 退回普通 /tts（不报错）",
        str(cosy_requests[0].url).endswith("/tts")
        and (Path(get_storage_root()) / "audio" / Path(plain_path).name).read_bytes() == b"normal",
        str(cosy_requests[0].url),
    )

    # 音色库里没有该音色 -> 也退回普通
    cosy_requests.clear()
    _install(cosy_handler)
    with engine.begin() as conn:
        run(tts.generate_tts(conn, {"text": "合成", "voice": "unknown-voice"}))
    check("cosy: 音色库查不到该 voice -> 退回普通 /tts", str(cosy_requests[0].url).endswith("/tts"))

    # ================= generate_voice_sample =================
    # 此时活跃的 audio 配置是 cosyvoice ⇒ mock 要同时认 /tts 与 /t2a_v2（就返回纯 hex 音频）
    def any_audio_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"base_resp": {"status_code": 0}, "data": {"audio": "aabb"}})

    _install(any_audio_handler)
    with engine.begin() as conn:
        sample_path = run(tts.generate_voice_sample(conn, "林昭", "v1"))
    check("voiceSample: 生成试听音频并返回路径", sample_path.startswith("static/audio/"), sample_path)
    with engine.begin() as conn:
        sample_usage = conn.execute(
            select(api_usage).order_by(api_usage.c.id.desc()).limit(1)
        ).first()
    check(
        "voiceSample: 试听文本按字符数计费（『你好，我是林昭。很高兴认识你，这是我的声音试听。』= 24 字）",
        sample_usage.units == 24, sample_usage.units,
    )

    # ================= voice_clone =================
    clone_seen: list[httpx.Request] = []

    def clone_handler(request: httpx.Request) -> httpx.Response:
        clone_seen.append(request)
        if str(request.url).endswith("/v1/files/upload"):
            return httpx.Response(200, json={"base_resp": {"status_code": 0},
                                             "file": {"file_id": 12345}})
        return httpx.Response(200, json={"base_resp": {"status_code": 0},
                                         "demo_audio": "https://cdn.test/demo.mp3"})

    _install(clone_handler)
    result = run(vc.clone_voice({
        "baseUrl": "https://api.minimax.test/",  # 结尾斜杠要能被剥掉
        "apiKey": "k",
        "fileBuffer": b"RIFF-sample",
        "filename": "sample.wav",
        "voiceId": "my-voice",
        "voiceName": "我的音色",
        "demoText": "试听文本",
    }))
    check(
        "clone: 返回 voiceId + demoAudio",
        result == {"voiceId": "my-voice", "demoAudio": "https://cdn.test/demo.mp3"}, result,
    )
    upload = clone_seen[0]
    check(
        "clone: 上传是 multipart（purpose=voice_clone + 文件名与 MIME）",
        upload.headers.get("content-type", "").startswith("multipart/form-data")
        and b"voice_clone" in upload.content and b"sample.wav" in upload.content
        and b"audio/wav" in upload.content,
        upload.headers.get("content-type"),
    )
    clone_body = json.loads(clone_seen[1].content)
    check(
        "clone: file_id 发成**数字**（`Number(fileId)`）、带降噪与响度归一",
        clone_body["file_id"] == 12345 and isinstance(clone_body["file_id"], int)
        and clone_body["need_noise_reduction"] is True
        and clone_body["need_volume_normalization"] is True,
        clone_body,
    )
    check(
        "clone: baseUrl 结尾斜杠被剥（不出现 //v1）",
        str(clone_seen[1].url) == "https://api.minimax.test/v1/voice_clone", str(clone_seen[1].url),
    )
    check(
        "clone: 有 demoText 时带上 text 与默认 model speech-2.8-hd",
        clone_body.get("text") == "试听文本" and clone_body.get("model") == "speech-2.8-hd", clone_body,
    )
    check(
        "clone: 有 voiceName 时带上 voice_name",
        clone_body.get("voice_name") == "我的音色", clone_body,
    )

    # 不带 demoText/voiceName 时不出现这些键
    clone_seen.clear()

    def clone_handler_plain(request: httpx.Request) -> httpx.Response:
        clone_seen.append(request)
        if str(request.url).endswith("/v1/files/upload"):
            return httpx.Response(200, json={"base_resp": {"status_code": 0}, "file": {"file_id": "7"}})
        return httpx.Response(200, json={"base_resp": {"status_code": 0}})

    _install(clone_handler_plain)
    plain = run(vc.clone_voice({
        "baseUrl": "https://api.minimax.test", "apiKey": "k",
        "fileBuffer": b"x", "filename": "a.mp3", "voiceId": "v",
    }))
    plain_body = json.loads(clone_seen[1].content)
    check(
        "clone: 无 demoText/voiceName 时不出现 text/model/voice_name 键；无 demo_audio 时是 None",
        "text" not in plain_body and "model" not in plain_body and "voice_name" not in plain_body
        and plain == {"voiceId": "v", "demoAudio": None},
        (plain_body, plain),
    )
    check("clone: file_id 字符串 '7' -> 数字 7", plain_body["file_id"] == 7, plain_body)

    # 上传失败 / 缺 file_id
    def upload_fail(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"base_resp": {"status_code": 1002, "status_msg": "样本太短"}})

    _install(upload_fail)
    try:
        run(vc.clone_voice({"baseUrl": "https://x", "apiKey": "k", "fileBuffer": b"x",
                            "filename": "a.mp3", "voiceId": "v"}))
        check("clone: 上传失败带 status_msg", False, "未抛错")
    except ValueError as err:
        check("clone: 上传失败用 status_msg 作为错误信息", str(err) == "样本太短", str(err))

    def upload_no_id(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"base_resp": {"status_code": 0}, "file": {}})

    _install(upload_no_id)
    try:
        run(vc.clone_voice({"baseUrl": "https://x", "apiKey": "k", "fileBuffer": b"x",
                            "filename": "a.mp3", "voiceId": "v"}))
        check("clone: 缺 file_id 报错", False, "未抛错")
    except ValueError as err:
        check("clone: 缺 file_id 报错", str(err) == "MiniMax 未返回 file_id", str(err))

    # 复刻失败：缺 base_resp 也算失败（undefined !== 0）
    def clone_no_base(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/v1/files/upload"):
            return httpx.Response(200, json={"base_resp": {"status_code": 0}, "file": {"file_id": 1}})
        return httpx.Response(200, json={})

    _install(clone_no_base)
    try:
        run(vc.clone_voice({"baseUrl": "https://x", "apiKey": "k", "fileBuffer": b"x",
                            "filename": "a.mp3", "voiceId": "v"}))
        check("clone: 复刻响应缺 base_resp 也判失败", False, "未抛错")
    except ValueError as err:
        check("clone: 复刻响应缺 base_resp 也判失败", str(err) == "MiniMax 音色复刻失败", str(err))

    # CosyVoice 零样本克隆
    cosy_clone_seen: list[httpx.Request] = []

    def cosy_clone_handler(request: httpx.Request) -> httpx.Response:
        cosy_clone_seen.append(request)
        return httpx.Response(200, json={"data": {"audio": "QUJD"}})

    _install(cosy_clone_handler)
    cosy_result = run(vc.clone_voice_cosyvoice({
        "baseUrl": "http://127.0.0.1:9880", "fileBuffer": b"RIFF", "promptText": "参考",
        "demoText": "试听",
    }))
    cosy_clone_body = json.loads(cosy_clone_seen[0].content)
    check(
        "cosyClone: 从 data.audio 取 demoAudio；prompt_audio 是 base64",
        cosy_result == {"demoAudio": "QUJD"}
        and cosy_clone_body["prompt_audio"] == base64.b64encode(b"RIFF").decode("ascii")
        and cosy_clone_body["model"] == "cosyvoice-v2",
        (cosy_result, {k: str(v)[:20] for k, v in cosy_clone_body.items()}),
    )

    # ================= 汇总 =================
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    # ⚠️ Windows 中文控制台是 **GBK**（py3.14 及以前不会自动 UTF-8 ✗）⇒ 不加这行，裸跑第一条 `print` 就 `UnicodeEncodeError` 崩 ✗（守口径见 `check_cli_encoding.py` ✓）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
