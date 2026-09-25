"""适配器层自检（``app/services/adapters/``）。

这一层**全是纯函数**（只构建请求描述与解析响应，自己不发 HTTP），所以可以完全离线、
逐个语义地把关。重点不在「能不能跑通」，而在**JS 语义是否逐字对齐** ——
URL 拼接、``parseInt``、``Math.round``、``??`` vs ``||``、``Buffer.from(x,'base64')``、
``JSON.stringify`` 丢 ``undefined``、``.length`` 对非数组是 undefined 等等。

运行::

    ./.venv/Scripts/python.exe tests/adapters_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="adapters_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.adapters import (  # noqa: E402
    UnknownProviderError,
    get_image_adapter,
    get_text_adapter,
    get_tts_adapter,
    get_video_adapter,
    join_provider_url,
)
from app.services.adapters.tts_adapters import map_emotion as _map_emotion  # noqa: E402
from app.services.voice_contract import EMOTION_ORDER as _EMOTION_ORDER  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _raises(fn, exc=ValueError):
    """执行并返回 (是否抛错, 异常信息)。"""
    try:
        fn()
    except exc as error:  # noqa: PERF203
        return True, str(error)
    return False, None


def main() -> int:  # noqa: C901  —— 自检脚本，平铺更直观
    # ================= join_provider_url =================
    check(
        "url: 裸 host + 前缀 + 路径",
        join_provider_url("http://x.com", "/v1", "/a") == "http://x.com/v1/a",
        join_provider_url("http://x.com", "/v1", "/a"),
    )
    check(
        "url: base 已含该前缀则不重复拼",
        join_provider_url("http://x.com/v1", "/v1", "/a") == "http://x.com/v1/a",
        join_provider_url("http://x.com/v1", "/v1", "/a"),
    )
    check(
        "url: base 带别的路径则接在后面",
        join_provider_url("http://x.com/api", "/v1", "/a") == "http://x.com/api/v1/a",
        join_provider_url("http://x.com/api", "/v1", "/a"),
    )
    check(
        "url: base 为空 -> 只拼前缀与路径",
        join_provider_url("", "/v1", "/a") == "/v1/a",
        join_provider_url("", "/v1", "/a"),
    )
    check(
        "url: 无 scheme 的 base 走 JS 的 catch 分支（字符串直拼）",
        join_provider_url("x.com", "/v1", "/a") == "x.com/v1/a",
        join_provider_url("x.com", "/v1", "/a"),
    )
    check(
        "url: 前缀为空 + 结尾斜杠 -> 不产生双斜杠",
        join_provider_url("http://x.com/", "", "/a") == "http://x.com/a",
        join_provider_url("http://x.com/", "", "/a"),
    )
    check(
        "url: scheme 与 host 转小写（URL 规范化）",
        join_provider_url("HTTP://X.COM", "/v1", "/a") == "http://x.com/v1/a",
        join_provider_url("HTTP://X.COM", "/v1", "/a"),
    )

    # ================= Gemini 图片：比例 / 档位 =================
    gemini = get_image_adapter("gemini")
    check(
        "gemini: 1920x1080 -> 16:9（gcd 归约，且不出现 16.0）",
        gemini._parse_aspect_ratio("1920x1080") == "16:9",
        gemini._parse_aspect_ratio("1920x1080"),
    )
    check("gemini: 1280x1280 -> 1:1", gemini._parse_aspect_ratio("1280x1280") == "1:1")
    check("gemini: 1080x1920 -> 9:16", gemini._parse_aspect_ratio("1080x1920") == "9:16")
    check("gemini: 非法 size -> 默认 16:9", gemini._parse_aspect_ratio("abc") == "16:9")
    check("gemini: 空 size -> 默认 16:9", gemini._parse_aspect_ratio(None) == "16:9")
    check(
        "gemini: 尺寸档位 2048->4K / 1024->2K / 512->1K / 384->512",
        [gemini._parse_image_size(s) for s in ("2048x1", "1024x1", "512x1", "384x1")]
        == ["4K", "2K", "1K", "512"],
        [gemini._parse_image_size(s) for s in ("2048x1", "1024x1", "512x1", "384x1")],
    )
    check("gemini: 无 size -> 1K", gemini._parse_image_size(None) == "1K")

    g_req = gemini.build_generate_request(
        {"baseUrl": "http://x.com", "apiKey": "k", "model": "gemini-2.5-flash-image"},
        {"prompt": "p", "size": "1920x1080"},
    )
    check("gemini: apiKey 同时进 query 与 header", "?key=k" in g_req["url"] and g_req["headers"]["x-goog-api-key"] == "k")
    check(
        "gemini: models/ 前缀自动补全",
        "models/gemini-2.5-flash-image:generateContent" in g_req["url"],
        g_req["url"],
    )
    check(
        "gemini: 无 prompt 时给默认文案",
        gemini.build_generate_request({}, {})["body"]["contents"][0]["parts"][-1]["text"] == "Generate an image",
    )
    check(
        "gemini: data URL 参考图 -> inline_data",
        gemini.build_generate_request(
            {"baseUrl": "http://x.com"}, {"prompt": "p", "referenceImages": '["data:image/png;base64,AAAB"]'}
        )["body"]["contents"][0]["parts"][0]
        == {"inline_data": {"mime_type": "image/png", "data": "AAAB"}},
    )
    check(
        "gemini: 非 data URL 的参考图被丢弃（parseDataUrl 返回 null）",
        len(
            gemini.build_generate_request(
                {"baseUrl": "http://x.com"}, {"prompt": "p", "referenceImages": '["http://a.png"]'}
            )["body"]["contents"][0]["parts"]
        )
        == 1,
    )

    # 安全拦截的归因文案
    raised, message = _raises(lambda: gemini.parse_generate_response(
        {"candidates": [{"finishReason": "IMAGE_SAFETY", "finishMessage": "blocked"}]}
    ))
    check(
        "gemini: 安全拦截 -> 中文可行动提示（不是英文 finishMessage）",
        raised and message is not None and message.startswith("图片生成被内容安全拦截"),
        message,
    )
    raised, message = _raises(lambda: gemini.parse_generate_response(
        {"candidates": [{"finishReason": "OTHER"}]}
    ))
    check(
        "gemini: 非安全类的 finishReason 原样带出",
        raised and message == "Gemini generation stopped: OTHER",
        message,
    )
    check(
        "gemini: STOP 不当作错误",
        gemini.parse_generate_response(
            {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"inlineData": {"data": "AA", "mimeType": "image/png"}}]}}]}
        )
        == {"isAsync": False, "imageUrl": None},
    )
    check(
        "gemini: 判定顺序是 URL 优先于 base64",
        gemini.parse_generate_response(
            {"data": [{"url": "u"}], "candidates": [{"content": {"parts": [{"inlineData": {"data": "AA"}}]}}]}
        )
        == {"isAsync": False, "imageUrl": "u"},
    )
    check(
        "gemini: inlineData 与 inline_data 两种写法都认",
        gemini.extract_image_base64({"candidates": [{"content": {"parts": [{"inline_data": {"data": "Z"}}]}}]})
        == {"data": "Z", "mimeType": "image/png"},
    )

    # ================= Ali 图片：尺寸映射 / seed =================
    ali_img = get_image_adapter("ali")
    check(
        "ali img: 宽幅 -> 1696*960 / 竖幅 -> 960*1696 / 方 -> 1280*1280",
        [ali_img._normalize_size(s) for s in ("1920x1080", "1080x1920", "1200x1200")]
        == ["1696*960", "960*1696", "1280*1280"],
        [ali_img._normalize_size(s) for s in ("1920x1080", "1080x1920", "1200x1200")],
    )
    check("ali img: 非法 size -> 1280*1280", ali_img._normalize_size("oops") == "1280*1280")
    ali_with_seed = ali_img.build_generate_request({"apiKey": "k"}, {"prompt": "p"})
    ali_no_seed = ali_img.build_generate_request({"apiKey": "k"}, {"prompt": "p", "referenceImages": '["u"]'})
    check(
        "ali img: 无参考图 -> 带随机 seed",
        "seed" in ali_with_seed["body"]["parameters"],
        ali_with_seed["body"]["parameters"],
    )
    check(
        "ali img: **有**参考图时不加 seed 键（对应 TS 的字面 undefined 被 JSON.stringify 丢掉）",
        "seed" not in ali_no_seed["body"]["parameters"],
        ali_no_seed["body"]["parameters"],
    )
    check(
        "ali img: 默认 baseUrl 是 dashscope",
        ali_img.build_generate_request({"apiKey": "k"}, {"prompt": "p"})["url"].startswith("https://dashscope.aliyuncs.com/api/v1/"),
    )
    check(
        "ali img: negative_prompt 缺省为空串（不是 null）",
        ali_with_seed["body"]["parameters"]["negative_prompt"] == "",
    )
    check(
        "ali img: 异步头 X-DashScope-Async=enable",
        ali_with_seed["headers"]["X-DashScope-Async"] == "enable",
    )
    check(
        "ali img: 未知响应把 JSON 截 200 字符带出（紧凑、不转义中文）",
        _raises(lambda: ali_img.parse_generate_response({"x": {"y": "中文"}}))[1]
        == 'Unexpected Ali image response: {"x":{"y":"中文"}}',
        _raises(lambda: ali_img.parse_generate_response({"x": {"y": "中文"}}))[1],
    )

    # ================= MiniMax / OpenAI / 火山 图片 =================
    minimax_img = get_image_adapter("minimax")
    check(
        "minimax img: body.size 有默认 1920x1080",
        minimax_img.build_generate_request({"apiKey": "k"}, {"prompt": "p"})["body"]["size"] == "1920x1080",
    )
    check(
        "minimax img: 传了 size 才派生 aspect_ratio",
        minimax_img.build_generate_request({"apiKey": "k"}, {"prompt": "p", "size": "1920x1080"})["body"]["aspect_ratio"]
        == "1920/1080",
    )
    # ⚠️ 反直觉但保真：`if (record.size)` 判的是**入参**，不是上面那个默认值 ⇒
    #    没传 size 时 body 里有 size 却没有 aspect_ratio（原 TS 就是这样）
    check(
        "minimax img: 未传 size 时**只有默认 size、没有 aspect_ratio**（判的是入参）",
        "aspect_ratio" not in minimax_img.build_generate_request({"apiKey": "k"}, {"prompt": "p"})["body"],
    )
    check(
        "minimax img: size 无 'x' 时不产 aspect_ratio（h 为 undefined）",
        "aspect_ratio" not in minimax_img.build_generate_request({"apiKey": "k"}, {"prompt": "p", "size": "1920"})["body"],
    )
    check(
        "minimax img: 参考图 JSON 用 record 覆盖，数组才吃",
        minimax_img.build_generate_request({"apiKey": "k"}, {"prompt": "p", "referenceImages": '["a","b"]'})["body"]["image"]
        == ["a", "b"],
    )
    check(
        "minimax img: referenceImages 是对象时不吃（JS 的 .length 为 undefined）",
        "image" not in minimax_img.build_generate_request({"apiKey": "k"}, {"prompt": "p", "referenceImages": '{"a":1}'})["body"],
    )
    check(
        "minimax img: 空响应报错文案",
        _raises(lambda: minimax_img.parse_generate_response({}))[1] == "No image URL or task_id in response",
        _raises(lambda: minimax_img.parse_generate_response({}))[1],
    )
    check(
        "minimax img: task_id 优先于图片 URL",
        minimax_img.parse_generate_response({"task_id": "t", "url": "u"}) == {"isAsync": True, "taskId": "t"},
    )
    check(
        "openai img: 默认 model 是 dall-e-3、默认 size 1024x1024",
        (lambda b: b["model"] == "dall-e-3" and b["size"] == "1024x1024")(
            get_image_adapter("openai").build_generate_request({"apiKey": "k"}, {})["body"]
        ),
    )
    check(
        "openai img: b64_json 模式 -> isAsync false 且不给 URL",
        get_image_adapter("openai").parse_generate_response({"data": [{"b64_json": "AA"}]})
        == {"isAsync": False, "imageUrl": None},
    )
    volc_img = get_image_adapter("volcengine")
    volc_parsed = volc_img.build_generate_request({"apiKey": "k"}, {"prompt": "p", "size": "1920abcx1080"})["body"]
    volc_nan = volc_img.build_generate_request({"apiKey": "k"}, {"prompt": "p", "size": "abcxdef"})["body"]
    # ⚠️ `parseInt('abc')` 在 JS 是 **NaN**（不是"不赋值"）⇒ 键仍在，只是 JSON 序列化成 null。
    #    写成「不设键」就与 Node 的请求体不一致了。
    check(
        "volc img: parseInt 取前导数字（'1920abc' -> 1920）",
        volc_parsed["width"] == 1920 and volc_parsed["height"] == 1080,
        volc_parsed,
    )
    check(
        "volc img: 解析不出时是 NaN -> None（键仍在，对应 JSON null）",
        "width" in volc_nan and volc_nan["width"] is None,
        volc_nan,
    )
    check(
        "volc img: 端点前缀 /api/v3",
        "/api/v3/images/generations" in volc_img.build_generate_request({"apiKey": "k"}, {})["url"],
    )

    # ================= 本地 SD =================
    sd = get_image_adapter("local-sd")
    check(
        "sd: 默认走 txt2img、无参考图",
        sd.build_generate_request({"baseUrl": "http://127.0.0.1:7860"}, {"prompt": "p"})["url"]
        == "http://127.0.0.1:7860/sdapi/v1/txt2img",
    )
    check(
        "sd: 有参考图 -> img2img + denoising_strength",
        (lambda b: b["url"].endswith("/sdapi/v1/img2img") and b["body"]["denoising_strength"] == 0.55)(
            sd.build_generate_request({"baseUrl": "http://127.0.0.1:7860"}, {"prompt": "p", "referenceImages": '["a"]'})
        ),
    )
    check(
        "sd: images 为空 -> 抛中文错误",
        _raises(lambda: sd.parse_generate_response({"images": []}))[1] == "SD 未返回图片",
    )
    check(
        "sd: images 是空对象**不**算空（JS 的 .length 为 undefined）",
        sd.parse_generate_response({"images": {}}) == {"isAsync": False},
    )
    check(
        "sd: 轮询接口一律抛错（同步模式）",
        _raises(lambda: sd.build_poll_request({}, "t"))[1] == "SD WebUI 同步模式，无需轮询"
        and _raises(lambda: sd.parse_poll_response({}))[1] == "SD WebUI 同步模式，无需轮询",
    )

    # ================= MiniMax 视频：checkpoint 路由 =================
    minimax_vid = get_video_adapter("minimax")
    cfg_h3 = {
        "apiKey": "k",
        "model": "base",
        "settings": {"checkpoint_map": {"fl2va": "F", "ref2va": "R"}},
    }
    check(
        "minimax vid: action/silent/transition/establishing/empty -> FL2VA",
        [minimax_vid.build_generate_request(cfg_h3, {"prompt": "p", "sceneType": s})["body"]["model"]
         for s in ("action", "SILENT", "transition shot", "establishing", "empty")] == ["F"] * 5,
    )
    check(
        "minimax vid: 其余（dialogue/meeting/...）-> Ref2VA",
        [minimax_vid.build_generate_request(cfg_h3, {"prompt": "p", "sceneType": s})["body"]["model"]
         for s in ("dialogue", "meeting", "argument", "long_dialogue", None)] == ["R"] * 5,
    )
    check(
        "minimax vid: 无 checkpoint_map -> record.model",
        minimax_vid.build_generate_request({"apiKey": "k", "model": "m"}, {"prompt": "p"})["body"]["model"] == "m",
    )
    check(
        "minimax vid: 多参考图 -> subject_reference 截断到 9 条",
        len(
            minimax_vid.build_generate_request(
                {"apiKey": "k"}, {"prompt": "p", "referenceMode": "multiple",
                                  "referenceImageUrls": "[" + ",".join(f'"{i}"' for i in range(12)) + "]"}
            )["body"]["subject_reference"]
        )
        == 9,
    )
    check(
        "minimax vid: 参考音频截断到 3 条",
        len(
            minimax_vid.build_generate_request(
                {"apiKey": "k"},
                {"prompt": "p", "referenceAudioUrls": '["a","b","c","d"]'},
            )["body"]["reference_audio"]
        )
        == 3,
    )
    check(
        "minimax vid: first_last 模式下两帧各就各位",
        (lambda b: b["first_frame_image"] == "f" and b["last_frame_image"] == "l")(
            minimax_vid.build_generate_request(
                {"apiKey": "k"}, {"prompt": "p", "referenceMode": "first_last",
                                  "firstFrameUrl": "f", "lastFrameUrl": "l"}
            )["body"]
        ),
    )

    # ================= 火山 / Vidu / Ali 视频 =================
    volc_vid = get_video_adapter("volcengine")
    check(
        "volc vid: duration 夹到 [4,12]（2->4 / 99->12 / 非数->5 / 7.5->8）",
        [volc_vid._normalize_duration(x) for x in (2, 99, "abc", 7.5)] == [4, 12, 5, 8],
        [volc_vid._normalize_duration(x) for x in (2, 99, "abc", 7.5)],
    )
    check(
        "volc vid: 首尾帧带 role 字段",
        (lambda c: c[1]["role"] == "first_frame" and c[2]["role"] == "last_frame")(
            volc_vid.build_generate_request(
                {"apiKey": "k"}, {"prompt": "p", "referenceMode": "first_last",
                                  "firstFrameUrl": "f", "lastFrameUrl": "l"}
            )["body"]["content"]
        ),
    )
    check(
        "volc vid: 默认 ratio 是 adaptive、generate_audio true",
        (lambda b: b["ratio"] == "adaptive" and b["generate_audio"] is True)(
            volc_vid.build_generate_request({"apiKey": "k"}, {"prompt": "p"})["body"]
        ),
    )
    vidu = get_video_adapter("vidu")
    check(
        "vidu: Authorization 用 Token 而不是 Bearer",
        vidu.build_generate_request({"apiKey": "k"}, {"prompt": "p"})["headers"]["Authorization"] == "Token k",
    )
    check(
        "vidu: images 恒存在（初始为 []），aspectRatio 映射成 resolution 720p",
        (lambda b: b["images"] == [] and b["resolution"] == "720p")(
            vidu.build_generate_request({"apiKey": "k"}, {"prompt": "p", "aspectRatio": "1:1"})["body"]
        ),
    )
    check(
        "vidu: 无轮询接口 -> 伪协议 URL + 永远 processing",
        vidu.build_poll_request({}, "t")["url"] == "vidu://no-polling-endpoint"
        and vidu.parse_poll_response({}) == {"status": "processing"},
    )
    from app.services.adapters.video_adapters import parse_callback_state

    check(
        "vidu: 回调 success/failed/未知 三态",
        parse_callback_state({"state": "success", "video_url": "v"}) == {"status": "completed", "videoUrl": "v"}
        and parse_callback_state({"state": "failed", "error": "e"}) == {"status": "failed", "error": "e"}
        and parse_callback_state({"state": "weird"}) == {"status": "failed", "error": "Unknown state: weird"},
    )
    ali_vid = get_video_adapter("ali")
    check(
        "ali vid: img_url 走 **nullish** 链（空串保留空串）",
        ali_vid.build_generate_request(
            {"apiKey": "k"}, {"prompt": "p", "imageUrl": "", "firstFrameUrl": "f"}
        )["body"]["input"]["img_url"]
        == "",
    )
    check(
        "ali vid: 两者都缺 -> 空串",
        ali_vid.build_generate_request({"apiKey": "k"}, {"prompt": "p"})["body"]["input"]["img_url"] == "",
    )
    check(
        "ali vid: resolution 映射 9:16/1:1 -> 720P，其余 1080P",
        [ali_vid._normalize_resolution(r) for r in ("9:16", "1:1", "16:9", None)] == ["720P", "720P", "1080P", "1080P"],
    )
    check(
        "ali vid: 尾帧写进 input.last_img_url",
        ali_vid.build_generate_request({"apiKey": "k"}, {"prompt": "p", "lastFrameUrl": "l"})["body"]["input"]["last_img_url"]
        == "l",
    )

    # ================= TTS =================
    minimax_tts = get_tts_adapter("minimax")
    check(
        "minimax tts: emotion 标签映射（furious->angry / excited->surprised / nervous->sad）",
        [minimax_tts.build_generate_request({"apiKey": "k"}, {"text": "x", "voice": "v", "emotion": e})["body"]
         ["voice_setting"]["emotion"] for e in ("furious", "EXCITED", "nervous")] == ["angry", "surprised", "sad"],
    )
    check(
        # ⚠️ 2026-09-24 改了**有意的行为** ✗（原来断言 ``|| 'happy'`` ✓✗）：未知情绪**当场拒** ✓，
        #    静默换成 happy =「用户选了难过、成片用开心念完、日志里一切正常」✓✗（见 voice_contract ✓）。
        "minimax tts: 未知 emotion -> **当场拒**（不再静默回落 happy ✗✗）",
        _raises(lambda: minimax_tts.build_generate_request(
            {"apiKey": "k"}, {"text": "x", "voice": "v", "emotion": "zzz"}))[0],
    )
    check(
        # ⭐ 不变量：契约的 8 个规范名**要么能映射、要么具名落在「表达不了」那侧** ✗
        #    （漏一个 ⇒ 用户选了却**生成到一半才炸** ✓✗ —— 正是这套接线要根除的形状 ✓）
        "minimax tts: 8 个规范名只有 disgust 没有落点（且它在能力声明里被具名排除 ✓）",
        [name for name in _EMOTION_ORDER if name not in minimax_tts.emotion_names] == ["disgust"],
    )
    check(
        "minimax tts: 声明「能表达」的名字**个个真能映射**（声明与表同源 ✓✗）",
        all(_map_emotion(name) for name in minimax_tts.emotion_names)
        and _raises(lambda: _map_emotion("disgust"))[0],
    )
    check(
        "minimax tts: speed/pitch 是 nullish（传 0 要保留 0）",
        (lambda v: v["speed"] == 0 and v["pitch"] == 0)(
            minimax_tts.build_generate_request(
                {"apiKey": "k"}, {"text": "x", "voice": "v", "speed": 0, "pitch": 0}
            )["body"]["voice_setting"]
        ),
    )
    check(
        "minimax tts: 不传 emotion 时不出现该键",
        "emotion" not in minimax_tts.build_generate_request({"apiKey": "k"}, {"text": "x", "voice": "v"})["body"]["voice_setting"],
    )
    check(
        "minimax tts: 缺 base_resp 也抛错（undefined !== 0）",
        _raises(lambda: minimax_tts.parse_response({"data": {"audio": "AA"}}))[1] == "TTS generation failed",
    )
    check(
        "minimax tts: status_code 非 0 用 status_msg",
        _raises(lambda: minimax_tts.parse_response({"base_resp": {"status_code": 1, "status_msg": "bad"}}))[1] == "bad",
    )
    check(
        "minimax tts: extra_info 缺省值补齐",
        minimax_tts.parse_response({"base_resp": {"status_code": 0}, "data": {"audio": "AA"}})
        == {"audioHex": "AA", "audioLength": 0, "sampleRate": 32000, "bitrate": 128000, "format": "mp3", "channel": 1},
    )
    cosy = get_tts_adapter("cosyvoice")
    check("cosy: 纯 hex 原样返回", cosy.parse_response({"audio": "deadbeef"})["audioHex"] == "deadbeef")
    check(
        "cosy: base64 自动转 hex",
        cosy.parse_response({"audio": "aGk="})["audioHex"] == "6869",
        cosy.parse_response({"audio": "aGk="})["audioHex"],
    )
    check(
        "cosy: 缺填充的 base64 也能解（Node Buffer 是宽容的）",
        cosy.parse_response({"audio": "aGk"})["audioHex"] == "6869",
        cosy.parse_response({"audio": "aGk"})["audioHex"],
    )
    check(
        "cosy: 无音频 -> 抛中文错误",
        _raises(lambda: cosy.parse_response({}))[1] == "CosyVoice 未返回音频数据",
    )
    check(
        "cosy: 带 promptAudio -> 走 /inference_zero_shot（tts_text/prompt_audio 字段）",
        (lambda r: r["url"].endswith("/inference_zero_shot") and r["body"]["tts_text"] == "x"
         and r["body"]["prompt_audio"] == "AAAA" and r["body"]["prompt_text"] == "")(
            cosy.build_generate_request({"baseUrl": "http://x.com"}, {"text": "x", "voice": "v", "promptAudio": "AAAA"})
        ),
    )
    check(
        "cosy: emotion 是 truthy 判断（传空串回退 neutral）",
        cosy.build_generate_request({"baseUrl": "http://x.com"}, {"text": "x", "voice": "v", "emotion": ""})["body"]["emotion"]
        == "neutral",
    )

    # ================= 文本 =================
    openai_text = get_text_adapter("openai")
    for provider, prefix in [
        ("openai", "/v1"),
        ("openrouter", "/v1"),
        ("chatfire", "/v1"),
        # ⚠️ `ollama` **不在此列**：2026-09-16 活体实测 —— 本机 qwen3 是思考模型，走 OpenAI 兼容的
        #    `/v1/chat/completions` 时 `message.content` **恒为空串**（正文在 `reasoning` 里）✗
        #    ⇒ ollama 已改用**原生** `/api/chat` + `think:false`（见 registry 与 text_adapters）。
        ("volcengine", "/api/v3"),
        ("ali", "/compatible-mode/v1"),
        ("minimax", "/v1"),
    ]:
        url = openai_text.build_request(
            {"provider": provider, "baseUrl": "http://x.com", "apiKey": "k"}, {"model": "m", "messages": []}
        )["url"]
        check(f"text: {provider} 前缀 {prefix}", url == f"http://x.com{prefix}/chat/completions", url)
    check(
        "text: 6 家共享同一个实例（不是复制体）",
        all(get_text_adapter(p) is openai_text
            for p in ("openai", "openrouter", "chatfire", "volcengine", "ali", "minimax")),
    )
    # ollama 走**原生**端点：/api/chat + think:false + options.num_predict（不是 OpenAI 形状）
    ollama_text = get_text_adapter("ollama")
    ollama_req = ollama_text.build_request(
        {"provider": "ollama", "baseUrl": "http://x.com"},
        {"model": "qwen3:14b", "messages": [{"role": "user", "content": "hi"}], "maxTokens": 8, "temperature": 0},
    )
    check("text: ollama 用原生 /api/chat（**不**走 OpenAI 兼容端点）",
          ollama_req["url"] == "http://x.com/api/chat", ollama_req["url"])
    check("text: ollama 关思考 + 上限字段名是 num_predict + temperature 保留 0",
          ollama_req["body"].get("think") is False and ollama_req["body"].get("stream") is False
          and ollama_req["body"]["options"] == {"temperature": 0, "num_predict": 8}, ollama_req["body"])
    check("text: ollama 解析原生 message.content",
          ollama_text.parse_response({"message": {"role": "assistant", "content": "收到"}}) == "收到")
    check(
        "text: temperature 是 nullish（传 0 保留 0）",
        openai_text.build_request({"provider": "openai"}, {"model": "m", "messages": [], "temperature": 0})["body"]["temperature"]
        == 0,
    )
    check(
        "text: maxTokens 为 0 时不带 max_tokens（truthy 判断）",
        "max_tokens" not in openai_text.build_request({"provider": "openai"}, {"model": "m", "messages": [], "maxTokens": 0})["body"],
    )
    check(
        "text: 数组 content 取第一段 text",
        openai_text.parse_response({"choices": [{"message": {"content": [{"type": "image_url"}, {"text": "A"}, {"text": "B"}]}}]})
        == "A",
    )
    check("text: content 缺失 -> 空串", openai_text.parse_response({"choices": []}) == "")
    gemini_text = get_text_adapter("gemini")
    g_body = gemini_text.build_request(
        {"baseUrl": "http://x.com", "apiKey": "k"},
        {"model": "gm", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                                     {"role": "assistant", "content": "a"}]},
    )["body"]
    check(
        "gemini text: system 抽到 systemInstruction、assistant 映射成 model",
        g_body["systemInstruction"] == {"parts": [{"text": "s"}]}
        and [c["role"] for c in g_body["contents"]] == ["user", "model"],
        g_body,
    )
    check(
        "gemini text: 无 system 时不出现 systemInstruction 键",
        "systemInstruction"
        not in gemini_text.build_request({"baseUrl": "http://x.com"}, {"model": "gm", "messages": [{"role": "user", "content": "u"}]})["body"],
    )
    check(
        "gemini text: maxTokens 写进 generationConfig.maxOutputTokens",
        gemini_text.build_request({"baseUrl": "http://x.com"}, {"model": "gm", "messages": [], "maxTokens": 64})["body"]
        ["generationConfig"]["maxOutputTokens"]
        == 64,
    )
    check(
        "gemini text: parts 全部拼接（不是取第一段）",
        gemini_text.parse_response({"candidates": [{"content": {"parts": [{"text": "A"}, {"text": "B"}]}}]}) == "AB",
    )
    check(
        "gemini text: 认证只有 x-goog-api-key（无 Authorization）",
        "Authorization"
        not in gemini_text.build_request({"baseUrl": "http://x.com", "apiKey": "k"}, {"model": "gm", "messages": []})["headers"],
    )

    # ================= 注册表 =================
    raised, message = _raises(lambda: get_image_adapter("nope"), UnknownProviderError)
    check(
        "registry: 未知 provider 抛错且列出可选项（不再静默回退 minimax）",
        raised and message is not None and message.startswith('Unknown image provider "nope". Available: ')
        and "local-sd" in message,
        message,
    )
    check(
        "registry: 大小写不敏感（provider 先 toLowerCase）",
        get_image_adapter("MiniMax") is get_image_adapter("minimax"),
    )
    check(
        "registry: chatfire 图片复用 OpenAI 实现类",
        type(get_image_adapter("chatfire")).__name__ == "OpenAIImageAdapter",
    )
    check(
        "registry: 四类未知 provider 的 kind 文案各自正确",
        _raises(lambda: get_video_adapter("x"), UnknownProviderError)[1].startswith("Unknown video provider")
        and _raises(lambda: get_tts_adapter("x"), UnknownProviderError)[1].startswith("Unknown TTS provider")
        and _raises(lambda: get_text_adapter("x"), UnknownProviderError)[1].startswith("Unknown text provider"),
    )

    # ================= 汇总 =================
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
