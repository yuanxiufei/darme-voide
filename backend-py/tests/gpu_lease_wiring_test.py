"""S7 自检：GPU 租约**接线**（text / tts 两处调用点）。

管理器本身在 `gpu_manager_test.py` 里锁过；这里锁的是**调用契约**（与 TS 逐条对齐）：

1. **只有本地配置**才申请租约（云端零开销）；
2. 申请参数是 ``(serviceType, provider, model, baseUrl)``，**serviceType 分别是 ``text`` / ``audio``**；
3. **每个模型尝试各自申请、``finally`` 里释放** —— 多模型 fallback 时是「申请→释放→申请→释放」，
   而不是一路持有（否则队列里后续请求要等整个 fallback 链结束）；
4. **异常路径也必须释放**（否则锁泄漏 ⇒ 后续所有本地请求永久排队）。

⚠️ 打桩两层：``gpu_manager``（记录申请/释放）+ ``fetch_with_retry``（不真发网络）。

运行::

    ./.venv/Scripts/python.exe tests/gpu_lease_wiring_test.py
"""
from __future__ import annotations

import asyncio
import importlib
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="lease_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ⚠️ Windows 控制台默认 GBK：检查名里带 emoji 时**打印阶段**会 UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.services import text_generation as tg  # noqa: E402
from app.services import tts_generation as tts  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_EVENTS: list[tuple] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


class _Lease:
    def __init__(self, key: str) -> None:
        self.modelKey = key

    def release(self) -> None:
        _EVENTS.append(("release", self.modelKey))


class _Recorder:
    """替身 gpu_manager：只记录申请，不发任何请求。"""

    async def acquire(self, service_type: str, provider: str, model: str,
                      base_url: str | None = None) -> _Lease:
        key = f"{service_type}:{model}"
        _EVENTS.append(("acquire", service_type, provider, model, base_url))
        return _Lease(key)


class _Response:
    def __init__(self, status: int = 200, payload: object = None, text: str = "") -> None:
        self.status_code = status
        self.is_success = 200 <= status < 300
        self.text = text
        self._payload = payload if payload is not None else {}

    def json(self) -> object:
        return self._payload


class _Adapter:
    """替身文本适配器：固定请求 + 固定解析结果。"""

    def build_request(self, config: dict, params: dict) -> dict:
        return {"url": "http://x/v1/chat", "method": "POST", "headers": {}, "body": {"m": params}}

    def parse_response(self, data: object) -> str:
        return "生成的文本"


def _text_config(provider: str, base_url: str, models: list[str]) -> dict:
    return {"provider": provider, "baseUrl": base_url, "models": models, "model": models[0],
            "apiKey": "k", "settings": json_dumps()}


def json_dumps() -> str:
    return "{}"


def main() -> int:  # noqa: C901
    recorder = _Recorder()
    tg.gpu_manager = recorder  # type: ignore[assignment]
    tts.gpu_manager = recorder  # type: ignore[assignment]
    tg.get_text_adapter = lambda _provider: _Adapter()  # type: ignore[assignment]

    async def run_text(provider: str, base_url: str, models: list[str],
                       outcomes: list[object]) -> object:
        """跑一次 generate_text，outcomes 逐模型给出「返回 _Response 或抛异常」。"""
        _EVENTS.clear()
        tg.get_text_config = lambda _conn: _text_config(provider, base_url, models)  # type: ignore[assignment]
        calls = {"n": 0}

        async def fake_fetch(url, request, service, **kwargs):
            index = min(calls["n"], len(outcomes) - 1)
            calls["n"] += 1
            outcome = outcomes[index]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        tg.fetch_with_retry = fake_fetch  # type: ignore[assignment]
        try:
            return await tg.generate_text(None, "提示词")
        except Exception as err:  # noqa: BLE001
            return err

    # 1) 本地配置：申请 + 释放各一次，参数齐全
    result = asyncio.run(run_text("ollama", "http://localhost:11434", ["qwen3:14b"],
                                  [_Response(payload={"ok": 1})]))
    check("本地: 文本生成成功，返回内容", result == "生成的文本", result)
    check("本地: 申请租约参数 = (text, provider, model, baseUrl)",
          ("acquire", "text", "ollama", "qwen3:14b", "http://localhost:11434") in _EVENTS,
          _EVENTS)
    check("本地: 成功路径也走 finally 释放（申请→释放各一次）",
          [e[0] for e in _EVENTS] == ["acquire", "release"], _EVENTS)

    # 2) 云端配置：完全不碰租约
    result = asyncio.run(run_text("chatfire", "https://api.chatfire.site", ["gemini-3-pro-preview"],
                                  [_Response(payload={"ok": 1})]))
    check("云端: 一次租约都不申请（零开销）", _EVENTS == [] and result == "生成的文本", _EVENTS)

    # 3) 本地 + fallback：第一个模型失败 ⇒ 每个尝试各自申请/释放（不跨模型持有）
    result = asyncio.run(run_text("ollama", "http://localhost:11434", ["qwen3:14b", "qwen3:8b"],
                                  [_Response(status=500, text="boom"),
                                   _Response(payload={"ok": 1})]))
    check("fallback: 第二个模型成功，返回内容", result == "生成的文本", result)
    check("fallback: 是「申请→释放→申请→释放」（不是一路持有）",
          [e[0] for e in _EVENTS] == ["acquire", "release", "acquire", "release"], _EVENTS)
    check("fallback: 两次申请分别是两个模型（第二个才成功）",
          [e[3] for e in _EVENTS if e[0] == "acquire"] == ["qwen3:14b", "qwen3:8b"], _EVENTS)

    # 4) 全失败：异常路径也必须释放（否则锁泄漏 ⇒ 后续本地请求永久排队）
    result = asyncio.run(run_text("ollama", "http://localhost:11434", ["qwen3:14b", "qwen3:8b"],
                                  [_Response(status=500, text="boom"),
                                   _Response(status=500, text="boom")]))
    check("异常: 全失败时抛出（不是静默返回空）", isinstance(result, Exception), repr(result))
    check("异常: 两次尝试都释放了（无锁泄漏）",
          [e[0] for e in _EVENTS] == ["acquire", "release", "acquire", "release"], _EVENTS)

    # 5) tts：本地申请（serviceType=audio）、失败也释放
    _EVENTS.clear()
    tts.get_audio_config_by_id = lambda _conn, _id: {  # type: ignore[assignment]
        "provider": "cosyvoice", "baseUrl": "http://localhost:9880",
        "models": ["cosyvoice-v2"], "model": "cosyvoice-v2", "settings": "{}"}

    async def boom_fetch(*_args, **_kwargs):
        raise RuntimeError("连接失败")

    tts.fetch_with_retry = boom_fetch  # type: ignore[assignment]
    try:
        asyncio.run(tts.generate_tts(None, {"text": "你好", "voice": "v1"}))
        check("tts: 失败应抛错", False)
    except Exception:  # noqa: BLE001
        check("tts: 失败照常抛错（不被租约逻辑吞掉）", True)
    check("tts: 本地配置申请的是 **audio** 租约（provider/model/baseUrl 齐全）",
          _EVENTS == [("acquire", "audio", "cosyvoice", "cosyvoice-v2", "http://localhost:9880"),
                      ("release", "audio:cosyvoice-v2")], _EVENTS)

    # ================= 长租约（image / video）=================
    ig = importlib.import_module("app.services.image_generation")
    vg = importlib.import_module("app.services.video_generation")

    for module, name, holder, key in (
        (ig, "image", ig, 4242), (vg, "video", vg, 4343),
    ):
        release = getattr(module, f"release_{name}_gpu_lease")
        leases = getattr(holder, f"_{name}_gpu_leases")
        hits: list[str] = []

        class _L:
            def release(self) -> None:
                hits.append("released")

        leases[key] = _L()
        release(key)
        check(f"{name}: 完成/失败时释放租约并**从表里删除**（幂等表，防重复释放）",
              hits == ["released"] and key not in leases, (hits, list(leases)))
        release(key)  # 再调一次
        check(f"{name}: 重复释放是 no-op（不会把别人的租约释放掉）", hits == ["released"], hits)
        release(999999)  # 不存在的 id
        check(f"{name}: 不存在的 id -> no-op（不抛）", hits == ["released"], hits)

    # 接线守卫：**四处（image）/ 三处（video）释放点**必须都在源码里
    # （少一处就是「锁泄漏 ⇒ 后续所有本地请求永久排队」，所以用机械检查钉住）
    src_ig = Path(ig.__file__).read_text(encoding="utf-8")
    src_vg = Path(vg.__file__).read_text(encoding="utf-8")
    check("image 接线: 重试前 / 末次失败 / 下载完成 / base64 完成 —— 四处都在",
          src_ig.count("release_image_gpu_lease(image_id)") == 4
          and "if attempt > 0:" in src_ig
          and "_image_gpu_leases[image_id] = await gpu_manager.acquire(" in src_ig,
          src_ig.count("release_image_gpu_lease(image_id)"))
    check("video 接线: 重试前 / 末次失败 / 完成 —— 三处都在",
          src_vg.count("release_video_gpu_lease(video_id)") == 3
          and "if attempt > 0:" in src_vg
          and "_video_gpu_leases[video_id] = await gpu_manager.acquire(" in src_vg,
          src_vg.count("release_video_gpu_lease(video_id)"))
    check("长租约: 申请**只对本地配置**发生（两处都以 is_local 为条件）",
          "if is_local:\n            try:\n                _image_gpu_leases[" in src_ig
          and "if is_local:\n            try:\n                _video_gpu_leases[" in src_vg)

    failed_items = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed_items)}/{len(_RESULTS)} passed")
    return 1 if failed_items else 0


if __name__ == "__main__":
    raise SystemExit(main())
