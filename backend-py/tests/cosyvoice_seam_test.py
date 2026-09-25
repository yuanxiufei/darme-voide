"""S7 自检：**CosyVoice 本地 TTS 的接缝契约**（2026-09-16 新增 —— 这条接缝此前从未被测过）。

背景（读上游源码核实，见 ``app/local_services/cosyvoice/server.py`` 的模块头表格）：
官方 FastAPI 服务与后端适配器**四项都对不上** ✗（无 ``/tts``、零样本要 multipart 文件、
字段叫 ``prompt_wav``、返回裸 PCM 流、端口 50000）⇒ 加了 ``cosyvoice/`` 薄封装把官方端点
包成后端期望的形状 ✓。本自检就是**把这层包装的服务契约钉死**：

    后端适配器（JSON） → 包装（``local_services/cosyvoice/server.py``）
                        → **官方端点形态**（form-data / multipart） → 裸 PCM → 补 WAV 头 → JSON hex

三层断言（都不需要真装 CosyVoice ✓）：

  A. **上游事实**：官方端点集合里**没有** ``/tts``（记录「为什么需要这层包装」✓）；包装声明了
     后端要用的三条路由 ✓；
  B. **真转发闭环**：起一个**真 HTTP 服务**当上游（uvicorn 线程，实现官方端点形态）⇒ 断言
     ① 包装把 JSON 映射成 form-data / multipart 打过去（字段名、文件、字节数都对 ✓）；
     ② 上游的裸 PCM 被补上**正确的 WAV 头**（RIFF/WAVE/data 长度、采样率、位深 ✓）；
     ③ ``CosyVoiceTTSAdapter.parse_response`` 能解析包装的响应 ✓（**适配器 ↔ 包装**契约闭环 ✓）；
     ④ **反套套逻辑**：上游命中计数必须 ≥1 —— 若包装压根没转发，这几条会一起失败 ✓（不会假绿 ✓）。
  C. **活体**（可选）：真上游（默认 50000）可达 ⇒ 真打一次 ``/inference_sft`` ✓；不可达 ⇒ 显式 SKIP ✓。

运行::

    ./.venv/Scripts/python.exe tests/cosyvoice_seam_test.py
"""
from __future__ import annotations

import base64
import importlib.util
import os
import socket
import struct
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="cosysvc_"))
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

import httpx  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, File, Form, UploadFile  # noqa: E402
from fastapi.responses import Response  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

#: 官方 FastAPI 服务（``runtime/python/fastapi/server.py``）**实际声明**的端点 —— 事实常量，
#: 改动前请重新核对上游源码（这份清单是「为什么不直接打官方端点」的依据 ✓）。
UPSTREAM_ENDPOINTS = (
    "/inference_sft", "/inference_zero_shot", "/inference_cross_lingual",
    "/inference_instruct", "/inference_instruct2",
)
#: 官方默认端口（后端 ``PRESET_SERVICES`` 里配的是包装的 9880 ✓，不是它 ✗）
UPSTREAM_DEFAULT_PORT = 50000
#: 上游返回的是**裸 int16 PCM**（无 WAV 头）—— 官方客户端自己硬编码 22050 Hz
UPSTREAM_SAMPLE_RATE = 22050

#: 假 PCM：0.05 秒 @22050 ⇒ 1102 采样 × 2 字节 = 2204 字节（形状明确、便于断言 ✓）
_FAKE_SAMPLES = 1102
_FAKE_PCM = struct.pack("<h", 1234) * _FAKE_SAMPLES

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []
#: stub 上游收到的请求（**反套套逻辑**的依据：没转发 ⇒ 这里就是空的 ✓）
HITS: dict[str, Any] = {}


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(name: str) -> None:
    _SKIPS.append(name)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_stub_upstream() -> FastAPI:
    """按**官方端点形态**造的 stub 上游（form-data / multipart / 裸 PCM 流）。

    ⚠️ 它必须**照官方来**：如果这里好心地收 JSON，那这份自检就测不出「包装有没有正确转换」✗。
    """
    stub = FastAPI()

    @stub.post("/inference_sft")
    async def sft(tts_text: str = Form(), spk_id: str = Form()) -> Response:  # noqa: D103
        HITS["sft"] = {"tts_text": tts_text, "spk_id": spk_id,
                       "content_type": "form"}
        return Response(content=_FAKE_PCM, media_type="application/octet-stream")

    @stub.post("/inference_zero_shot")
    async def zero_shot(  # noqa: D103
        tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()
    ) -> Response:
        raw = await prompt_wav.read()
        HITS["zero_shot"] = {"tts_text": tts_text, "prompt_text": prompt_text,
                             "prompt_wav_name": prompt_wav.filename, "prompt_wav_bytes": len(raw),
                             "content_type": "multipart"}
        return Response(content=_FAKE_PCM, media_type="application/octet-stream")

    return stub


def start_server(app: FastAPI) -> tuple[str, uvicorn.Server, threading.Thread]:
    """在后台线程里起真 HTTP 服务（真 socket ⇒ 包装的 httpx 能真发 ✓）。"""
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 25
    while time.time() < deadline:
        try:
            if httpx.get(f"{base}/docs", timeout=1).status_code == 200:
                return base, server, thread
        except Exception:  # noqa: BLE001
            time.sleep(0.1)
    raise RuntimeError("stub 上游没能起来")


#: ⚠️ 包装源码**不在本仓跟踪范围** ✓✗（`.gitignore`：`backend-py/app/local_services/*` 只留 ``h3/`` ✓）
#: ⇒ 干净的检出里它**不存在** ✓ ⇒ 这套必须"没就位就显式跳过" ✗（见 :func:`main`）。
WRAPPER_PATH = BACKEND_PY / "app" / "local_services" / "cosyvoice" / "server.py"


def load_wrapper() -> Any:
    """按路径加载包装（它在 ``app/local_services/`` 下，**不是**应用分层代码，故按文件加载 ✓）。"""
    path = WRAPPER_PATH
    spec = importlib.util.spec_from_file_location("cosyvoice_wrapper_server", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def parse_wav(data: bytes) -> dict[str, Any]:
    """极简 WAV 头解析（只读标准 44 字节头 ✓）。"""
    return {
        "riff": data[0:4],
        "wave": data[8:12],
        "fmt_tag": data[12:16],
        "channels": struct.unpack("<H", data[22:24])[0],
        "sample_rate": struct.unpack("<I", data[24:28])[0],
        "bits": struct.unpack("<H", data[34:36])[0],
        "data_tag": data[36:40],
        "data_size": struct.unpack("<I", data[40:44])[0],
    }


def main() -> int:  # noqa: C901
    # ⚠️ **没就位就显式跳过** ✗（2026-09-25 实测抓到 ✓）：包装源码是**本地部署产物** ✓（不在仓里 ✓），
    #   原来的 `spec_from_file_location(None)` 会直接抛 `FileNotFoundError` ⇒ **整套崩** ⇒
    #   连"这条接缝这次根本没验"都看不见 ✗（比红更难查 ✗）⇒ 现在明说。
    if not WRAPPER_PATH.exists():
        skip(f"包装源码未就位（{WRAPPER_PATH.relative_to(BACKEND_PY)}）⇒ A/B 两组都验不了"
             f"（C 组是活体、本就要上游 ✓）")
        for name in _SKIPS:
            print("SKIP  " + name + "（源码未就位 ⇒ 该套按设计未验证，**不是通过**）")
        print()
        print("SUMMARY: 0/0 passed（skip 1：本源码未就位）")
        return 0
    base, server, thread = start_server(build_stub_upstream())
    try:
        # ⚠️ 包装在**模块级**读 COSYVOICE_UPSTREAM ⇒ 必须先设环境变量再加载 ✓
        os.environ["COSYVOICE_UPSTREAM"] = base
        module = load_wrapper()
        client = TestClient(module.app)

        # ── A. 上游事实 + 包装路由表 ──
        check("A① 上游事实: 官方端点里**没有** /tts（这就是本层包装存在的理由）",
              "/tts" not in UPSTREAM_ENDPOINTS, UPSTREAM_ENDPOINTS)
        declared = {getattr(route, "path", "") for route in module.app.routes}
        check("A② 包装: 声明了后端要用的三条路由（/tts、/inference_zero_shot、/healthz）",
              {"/tts", "/inference_zero_shot", "/healthz"} <= declared, sorted(declared))
        check("A③ 包装: 上游地址取自环境变量（可指到真服务或 stub）",
              module.UPSTREAM == base, module.UPSTREAM)

        # ── B. 普通合成：JSON → form-data → 裸 PCM → WAV → hex ──
        plain = client.post("/tts", json={
            "text": "你好，这是本地合成。", "voice_id": "女声A",
            # 下面这几个字段后端一直在送，但官方 SFT 端点**都不认** ⇒ 包装应忽略并如实回 wav ✓
            "emotion": "happy", "format": "mp3", "sample_rate": 32000, "model": "cosyvoice-v2",
        })
        check("B① /tts 返回 200（JSON 进、包装自己转形态）", plain.status_code == 200, plain.text[:200])
        body = plain.json() if plain.status_code == 200 else {}
        hit = HITS.get("sft") or {}
        check("B② ⭐ 上游收到 **form-data** 且字段名正确（text→tts_text、voice_id→spk_id）",
              hit.get("content_type") == "form" and hit.get("tts_text") == "你好，这是本地合成。"
              and hit.get("spk_id") == "女声A", hit)
        check("B③ 反套套逻辑: 上游命中计数存在（包装**真的转发了**，不是本地造了个响应）",
              "sft" in HITS, list(HITS))

        wav_bytes = bytes.fromhex(body.get("audio") or "")
        wav = parse_wav(wav_bytes) if len(wav_bytes) >= 44 else {}
        check("B④ 响应是 **hex**（无 base64/hex 歧义 ⇒ 适配器 is_pure_hex 判定不会误判）",
              bool(body.get("audio")) and all(c in "0123456789abcdef" for c in str(body["audio"])[:200]),
              str(body.get("audio"))[:40])
        check("B⑤ 包装补的 WAV 头正确（RIFF/WAVE/fmt/data + 采样率 22050 + 16bit 单声道）",
              wav.get("riff") == b"RIFF" and wav.get("wave") == b"WAVE"
              and wav.get("fmt_tag") == b"fmt " and wav.get("data_tag") == b"data"
              and wav.get("sample_rate") == UPSTREAM_SAMPLE_RATE and wav.get("bits") == 16
              and wav.get("channels") == 1, wav)
        check("B⑥ WAV 里装的就是上游那段 PCM（data 长度 == 上游 PCM 字节数）",
              wav.get("data_size") == len(_FAKE_PCM)
              and len(wav_bytes) == 44 + len(_FAKE_PCM), (wav.get("data_size"), len(wav_bytes)))
        check("B⑦ 如实回报 format=wav（下游用它当**文件扩展名** ⇒ 不能沿用后端的 mp3 假设）",
              body.get("format") == "wav", body.get("format"))

        # ── B'. 适配器 ↔ 包装 契约闭环 ──
        from app.services.adapters.tts_adapters import CosyVoiceTTSAdapter

        parsed = CosyVoiceTTSAdapter().parse_response(body)
        check("B⑧ 适配器能解析包装的响应（audioHex 非空 + format=wav + sampleRate 对齐）",
              bool(parsed.get("audioHex")) and parsed.get("format") == "wav"
              and parsed.get("sampleRate") == UPSTREAM_SAMPLE_RATE, parsed)

        # ── B''. 默认说话人 + 零样本（base64 → multipart 文件） ──
        client.post("/tts", json={"text": "默认音色"})
        check("B⑨ 未给 voice_id ⇒ 用默认说话人（官方客户端同款 '中文女'）",
              (HITS.get("sft") or {}).get("spk_id") == "中文女", HITS.get("sft"))

        prompt_pcm = b"RIFFpcm-sample-bytes" * 8
        zero = client.post("/inference_zero_shot", json={
            "tts_text": "零样本合成文本",
            "prompt_text": "参考音频对应文本",
            "prompt_audio": base64.b64encode(prompt_pcm).decode("ascii"),
            "model": "cosyvoice-v2", "stream": False, "speed": 1.0,
        })
        zhit = HITS.get("zero_shot") or {}
        check("B⑩ /inference_zero_shot 返回 200（JSON + base64 进）", zero.status_code == 200, zero.text[:200])
        check("B⑪ ⭐ 上游收到 **multipart 文件** prompt_wav，且字节数 == 解码后的 base64 字节数",
              zhit.get("content_type") == "multipart" and zhit.get("prompt_wav_bytes") == len(prompt_pcm)
              and bool(zhit.get("prompt_wav_name")), zhit)
        check("B⑫ 零样本的两个文本字段也按官方名传（tts_text / prompt_text）",
              zhit.get("tts_text") == "零样本合成文本"
              and zhit.get("prompt_text") == "参考音频对应文本", zhit)

        # ── B'''. 错误面：坏 base64 / 空文本 / 上游不可达 ──
        check("B⑬ 空 text ⇒ 400（不乱打上游）", client.post("/tts", json={"text": "  "}).status_code == 400)
        check("B⑭ 坏 base64 的 prompt_audio ⇒ 400（带原因，不吞错）",
              client.post("/inference_zero_shot",
                          json={"tts_text": "x", "prompt_audio": "!!!not-base64!!!"}).status_code == 400)
        health = client.get("/healthz").json()
        check("B⑮ /healthz 找到上游（upstream_reachable=True ⇒ 排障时一眼看出上游起没起）",
              health.get("ok") is True and health.get("upstream_reachable") is True, health)

        # ── C. 活体（真 CosyVoice 服务，默认 50000）：可达就真打，否则显式 SKIP ──
        real = f"http://127.0.0.1:{UPSTREAM_DEFAULT_PORT}"
        try:
            alive = httpx.get(f"{real}/docs", timeout=3).status_code == 200
        except Exception:  # noqa: BLE001
            alive = False
        if alive:
            started = time.time()
            live = httpx.post(f"{real}/inference_sft",
                              data={"tts_text": "活体探测", "spk_id": "中文女"}, timeout=180)
            check("C① 活体: 真上游 /inference_sft 返回 2xx 且响应体非空（裸 PCM）",
                  live.status_code // 100 == 2 and len(live.content) > 0,
                  (live.status_code, len(live.content), round(time.time() - started, 1)))
        else:
            skip(f"真 CosyVoice 上游（{UPSTREAM_DEFAULT_PORT}）未在跑 ⇒ C 组按设计未验证"
                 f"（B 组已用真 HTTP stub 覆盖转换逻辑 ✓）")
    finally:
        server.should_exit = True
        thread.join(timeout=5)

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    for name in _SKIPS:
        print("SKIP  " + name + "（服务未启动 ⇒ 该条按设计未验证，**不是通过**）")
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed"
          + (f"（skip {len(_SKIPS)}：服务未启动）" if _SKIPS else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
