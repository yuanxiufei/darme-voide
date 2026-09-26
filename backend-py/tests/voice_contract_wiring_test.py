"""自检：**配音情绪/语速契约的接线**（2026-09-24 ✓，零网络 ✓）。

对象是「能力接不出去不算功能」✗ 那条：``voice_contract`` 的判据此前**没有任何调用方** ✗✗
（全仓 0 引用 ✓），而生产链上躺着两处**静默改/丢用户意图** ✗✗：

* MiniMax 适配器 ``_MINIMAX_EMOTION_MAP.get(e) or 'happy'`` ✓✗ —— 未知情绪**静默变成开心** ✓；
* CosyVoice 两条路径（``/tts`` → ``/inference_sft`` ✓、零样本 ✓）**都不转发** ``emotion``/``speed`` ✗✗
  —— 用户选的情绪**静默消失** ✓（包装器自己写着「明确忽略」✓）。

本套钉的判据（错了都是「出片不对还查不出来」✓✗）：

1. ⭐⭐ **非法 ⇒ 提交前拒** ✗：``generate_tts`` 在**发请求之前**就抛 ✓（一个请求都不发 ✓✗）；
2. ⭐⭐ **引擎不认 ⇒ 响亮报告** ✗：进 ``dropped`` + 理由 ✓（**不静默丢** ✓）且**不因此让整单失败** ✓；
3. ⭐⭐ **「没给」不是「默认」** ✗：``None``/``""`` ⇒ 不带情绪 ✓（**不替你选 happy** ✓✗）；
4. ⭐ **「没查」不是「通过」** ✗：语速区间三处来源都没有 ⇒ ``speed_range_checked=False`` ✓ 且**如实报告** ✓✗。

⚠️ 与 ``voice_contract_test.py`` 分工：那套只验**纯契约**（零依赖 ✓），本套验**接线**（真跑
``generate_tts`` + 真 HTTP stub ✓ 证明「拦在提交前」✓）。

运行::

    ./.venv/Scripts/python.exe tests/voice_contract_wiring_test.py
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="vccontract_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import get_storage_root  # noqa: E402
from app.core.db import engine  # noqa: E402
from app.core.models import ai_voices  # noqa: E402
from app.core.response import now  # noqa: E402
from app.main import app  # noqa: E402
from app.services import tts_generation as tts  # noqa: E402
from app.services import vendor_errors  # noqa: E402
from app.services import voice_contract as vc  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def run(coro):
    return asyncio.run(coro)


def raises_contract(fn, *args: object, **kwargs: object) -> str | None:
    """执行并返回**拒绝理由** ✓（没拒 ⇒ ``None`` ✓ —— 用它当判据比 bool 更能钉住「拒对了没」✓）。"""
    try:
        fn(*args, **kwargs)
    except vc.VoiceContractError as err:
        return str(err)
    return None


class FakeAdapter:
    """假适配器 ✓（**只**声明能力 ✓）：接线判据不依赖真适配器 ✓（改厂商映射不该打碎这套 ✓✗）。"""

    def __init__(self, **declared: object) -> None:
        for key, value in declared.items():
            setattr(self, key, value)


# ══════════════════════════════════════════════════════════════════════════
# ① 契约层：规范值 / 拒绝 / 报告（零依赖 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_resolve_emotion() -> None:
    full = FakeAdapter(supports_emotion=True, supports_emotion_vector=False,
                       emotion_names=vc.EMOTION_ORDER, supports_speed=True,
                       speed_range=(0.5, 2.0))

    plan = vc.resolve_voice_params({"emotion": "开心", "speed": 1.2}, adapter=full)
    check("① 中文别名 ⇒ 规范名 happy ✓（**同义映射**不是猜最近邻 ✗）",
          plan["emotion"] == "happy" and plan["emotion_mode"] == "preset"
          and plan["emotion_provided"] is True, plan)
    check("①′ 语速有区间 ⇒ 原值 + ``speed_range_checked=True`` ✓（来源如实标注 ✓）",
          plan["speed"] == 1.2 and plan["speed_range_checked"] is True
          and plan["declared"]["speed_range_source"] == "adapter", plan["declared"])

    check("② ⭐⭐ 未知情绪 ⇒ **拒**（不再静默变 happy ✗✗）",
          raises_contract(vc.resolve_voice_params, {"emotion": "zzz"}, adapter=full),
          vc.resolve_voice_params({"emotion": "happy"}, adapter=full))
    check("②′ 向量 + 引擎收不了向量 ⇒ **拒**（丢的是**整条意图** ✗✗ 不许静默）",
          "收不了情绪向量" in (raises_contract(vc.resolve_voice_params, {"emotion": [0.1] * 8},
                                               adapter=full) or ""))

    absent = vc.resolve_voice_params({"emotion": "", "speed": None}, adapter=full)
    check("③ ⭐⭐ ``''``/``None`` ⇒ **不带情绪**，且**不替用户默认** ✗（≠ happy ✓✗）",
          absent["emotion"] is None and absent["emotion_provided"] is False
          and absent["speed"] is None and absent["speed_provided"] is False, absent)
    check("③′ ⚠️ 「没给」是**最常见**的情况 ⇒ **不写 note** ✗（写了会把真问题淹掉 ✓✗）",
          absent["notes"] == [] and absent["dropped"] == [], absent["notes"])

    vector_ok = vc.resolve_voice_params({"emotion": [0.0, 1.0, 0, 0, 0, 0, 0, 0]},
                                        adapter=FakeAdapter(supports_emotion=True,
                                                            supports_emotion_vector=True,
                                                            emotion_names=vc.EMOTION_ORDER))
    check("③″ 引擎收向量 ⇒ 出 8 维列表 + ``mode=vector`` ✓（顺序恒为契约序 ✓）",
          vector_ok["emotion_mode"] == "vector" and len(vector_ok["emotion"]) == 8
          and vector_ok["emotion"][1] == 1.0, vector_ok["emotion"])


def case_capability_report() -> None:
    check("④ ⭐⭐ 引擎**表达不了**某个情绪 ⇒ 进 ``dropped`` 且**点名**（不猜最近邻 ✗）",
          (lambda p: p["emotion"] is None and len(p["dropped"]) == 1
                     and "表达不了" in p["dropped"][0]["reason"])(
              vc.resolve_voice_params({"emotion": "disgust"},
                                      adapter=FakeAdapter(supports_emotion=True,
                                                          emotion_names=("happy", "sad")))))

    check("④′ ⭐ ``supports_emotion=False`` ⇒ 进 ``dropped``、理由说清「不接收情绪」✓",
          (lambda p: p["emotion"] is None and "不接收情绪" in p["dropped"][0]["reason"]
                     and p["dropped"][0]["field"] == "emotion")(
              vc.resolve_voice_params({"emotion": "happy"},
                                      adapter=FakeAdapter(supports_emotion=False))))

    unknown = vc.resolve_voice_params({"emotion": "happy"}, adapter=FakeAdapter())
    check("⑤ ⭐⭐ 能力**没声明** ⇒ 记「没查」✗ 且**不当作支持** ✗✗（原样传出 ✓ 不猜 ✓）",
          unknown["emotion"] == "happy" and unknown["declared"]["supports_emotion"] is None
          and any("没查" in note for note in unknown["notes"]), unknown["notes"])
    check("⑤′ 「没查」与「支持」的**报告不同** ✗（否则「没查」会被读成绿灯 ✓✗）",
          vc.resolve_voice_params({"emotion": "happy"},
                                  adapter=FakeAdapter(supports_emotion=True))["notes"] == []
          and unknown["notes"] != [])
    nope = vc.resolve_voice_params({"emotion": "happy"}, adapter=None)
    check("⑤″ 连适配器都没有 ⇒ 声明全 ``None``、**不当通过** ✗（这里不负责报未知 provider ✗）",
          nope["declared"]["supports_emotion"] is None
          and nope["declared"]["supports_speed"] is None and nope["emotion"] == "happy")


def case_capability_speed() -> None:
    no_range = vc.resolve_voice_params({"speed": 5.0},
                                       adapter=FakeAdapter(supports_speed=True))
    check("⑥ ⭐⭐ 区间**三处来源都没有** ⇒ 只做有限性校验 ✓ 且 ``speed_range_checked=False`` ✓✗"
          "（**「没查」不是「通过」** ✗ —— 5.0 在这里放行但被如实标成没核 ✓）",
          no_range["speed"] == 5.0 and no_range["speed_range_checked"] is False
          and any("区间没查" in note for note in no_range["notes"]), no_range["notes"])
    check("⑥′ 有区间时**同一个值** ⇒ 拒 ✓（证明上面那条不是「管不了」✗ 而是「没核」✓）",
          raises_contract(vc.resolve_voice_params, {"speed": 5.0},
                          adapter=FakeAdapter(supports_speed=True, speed_range=(0.5, 2.0))))
    check("⑥″ 调用方给的区间**优先于**适配器声明 ✓（离引擎最近的那份说了算 ✓）",
          (lambda p: p["declared"]["speed_range_source"] == "caller"
                     and p["speed_range_checked"] is True)(
              vc.resolve_voice_params({"speed": 3.0}, adapter=FakeAdapter(speed_range=(0.5, 2.0)),
                                      speed_range=(0.5, 4.0))))
    check("⑥‴ 越界 ⇒ **拒**（**不悄悄钳位** ✓✗）；非数值 ⇒ 拒 ✓；半区间 ⇒ 拒 ✓（两个都给才对 ✓）",
          raises_contract(vc.resolve_voice_params, {"speed": 3.0},
                          adapter=FakeAdapter(speed_range=(0.5, 2.0)))
          and raises_contract(vc.resolve_voice_params, {"speed": "abc"},
                              adapter=FakeAdapter(speed_range=(0.5, 2.0)))
          and raises_contract(vc.validate_speed, 1.0, low=0.5))
    dropped_speed = vc.resolve_voice_params({"speed": 1.2}, adapter=FakeAdapter(supports_speed=False))
    check("⑥⁗ 引擎不转发语速 ⇒ ``dropped`` + **不还给调用方** ✓（speed=None ✓）",
          dropped_speed["speed"] is None and "不转发语速" in dropped_speed["dropped"][0]["reason"],
          dropped_speed)
    clone_plan = vc.resolve_voice_params({"emotion": "happy"}, clone=True,
                                         adapter=FakeAdapter(supports_emotion=False))
    check("⑦ ⭐ 克隆路径的理由**要点名当前是克隆** ✓（否则用户读不出为什么情绪没生效 ✓✗）",
          "克隆" in clone_plan["dropped"][0]["reason"], clone_plan["dropped"])


# ══════════════════════════════════════════════════════════════════════════
# ② 接线层：真跑 generate_tts（HTTP 全 mock ✓、**一个真网络都没有** ✗）
# ══════════════════════════════════════════════════════════════════════════
def case_generate_tts_wiring() -> None:
    client = TestClient(app)
    client.post("/api/v1/ai-configs", json={
        "service_type": "audio", "provider": "minimax", "base_url": "https://api.minimax.test",
        "api_key": "k", "model": ["speech-2.8-hd"], "is_active": True,
    })

    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={
            "base_resp": {"status_code": 0},
            "data": {"audio": "6869", "extra_info": {"audio_length": 100, "audio_format": "mp3"}},
        })

    # 捕获任务日志 ✓：接线的「报告」就落在这里 ✓ ⇒ 抓它才算证明**真跑过** ✓✗
    logged: list[tuple[str, str, dict]] = []
    original = (tts.log_task_payload, tts.log_task_warn, tts.log_task_error)
    tts.log_task_payload = lambda cat, action, meta=None: logged.append(("payload", action, meta or {}))
    tts.log_task_warn = lambda cat, action, meta=None: logged.append(("warn", action, meta or {}))
    tts.log_task_error = lambda cat, action, meta=None: logged.append(("error", action, meta or {}))
    try:
        vendor_errors._vendor_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        # ── ① 非法情绪：**拦在提交前** ✗✗（一个请求都不许发 ✓）──
        rejected: str | None = None
        with engine.begin() as conn:
            try:
                run(tts.generate_tts(conn, {"text": "你好", "voice": "v1", "emotion": "zzz"}))
            except vc.VoiceContractError as err:
                rejected = str(err)
        check("⑧ ⭐⭐ 非法情绪 ⇒ ``generate_tts`` 抛 **VoiceContractError** ✓（提交前 ✓）",
              bool(rejected) and "不认识的情绪" in str(rejected), rejected)
        check("⑧′ ⭐⭐ 而且**一个厂商请求都没发出去** ✗✗（这才是「拦在提交前」的判据 ✓）",
              sent == [], [str(r.url) for r in sent])
        check("⑧″ 拒绝也**要留痕** ✓（否则前端只看到一个 500 ✓✗）",
              any(action == "voice-contract-rejected" for _, action, _ in logged), logged[:2])

        # ── ② 合法值：规范名进请求体 ✓ ──
        sent.clear()
        logged.clear()
        with engine.begin() as conn:
            run(tts.generate_tts(conn, {"text": "你好", "voice": "v1", "speed": 1.2,
                                        "emotion": "开心"}))
        body = json.loads(sent[0].content)["voice_setting"] if sent else {}
        check("⑨ 接线后：别名被规范化后才进请求体 ✓（``开心`` ⇒ ``happy`` ✓）",
              body.get("emotion") == "happy" and body.get("speed") == 1.2, body)
        contract_log = [meta for kind, action, meta in logged if action == "voice contract"]
        check("⑨′ 契约报告**落进任务日志** ✓（``speed_range_checked`` 也在里面 ✓✗ —— "
              "「没查」必须能被查出来 ✓）",
              len(contract_log) == 1 and contract_log[0]["speed_range_checked"] is False
              and contract_log[0]["declared"]["speed_range"] is None, contract_log)

        # ── ③ 引擎表达不了：**响亮报告**但不拖垮整单 ✓ ──
        sent.clear()
        logged.clear()
        with engine.begin() as conn:
            path = run(tts.generate_tts(conn, {"text": "你好", "voice": "v1", "emotion": "disgust"}))
        dropped_calls = [meta for kind, action, meta in logged if action == "voice-contract-dropped"]
        check("⑩ ⭐ 引擎表达不了 ⇒ **不失败** ✓（生成照旧成功 ✓）但**必须报** ✗",
              path.startswith("static/audio/") and len(dropped_calls) == 1
              and dropped_calls[0]["field"] == "emotion", (path, dropped_calls))
        check("⑩′ 丢掉的值**不出现在请求体**里 ✓（不是「发了但被忽略」那种假接线 ✗）",
              "emotion" not in json.loads(sent[0].content)["voice_setting"],
              json.loads(sent[0].content)["voice_setting"])

        # ── ④ 配置里给了区间 ⇒ 越界当场拒 ✓（证明 ``settings.speedRange`` 真被读了 ✓）──
        client.post("/api/v1/ai-configs", json={
            "service_type": "audio", "provider": "minimax", "base_url": "https://api.minimax.test",
            "api_key": "k", "model": ["speech-2.8-hd"], "is_active": True, "priority": 9,
            "settings": {"speedRange": [0.5, 2.0]},
        })
        sent.clear()
        with engine.begin() as conn:
            try:
                run(tts.generate_tts(conn, {"text": "你好", "voice": "v1", "speed": 5.0}))
                check("⑪ 配置里的语速区间**真被读进接线** ✓（没读 ⇒ 这条会静默放行 ✓✗）",
                      False, "没抛错")
            except vc.VoiceContractError as err:
                check("⑪ 配置里的语速区间**真被读进接线** ✓（越界 ⇒ 提交前拒 ✓）",
                      "不在 [0.5, 2.0] 内" in str(err) and sent == [], (str(err), len(sent)))

        # ── ⑤ 本地引擎：克隆路径**不静默丢** ✗✗ ──
        storage = Path(get_storage_root())
        (storage / "audio").mkdir(parents=True, exist_ok=True)
        (storage / "audio" / "ref.wav").write_bytes(b"RIFF-fake")
        with engine.begin() as conn:
            values = {"voice_id": "cosy-1", "voice_name": "音色", "provider": "cosyvoice",
                      "reference_audio": "static/audio/ref.wav", "prompt_text": "参考文本"}
            if "created_at" in ai_voices.c:
                values["created_at"] = now()
            conn.execute(ai_voices.insert().values(**values))
        client.post("/api/v1/ai-configs", json={
            "service_type": "audio", "provider": "cosyvoice", "base_url": "http://127.0.0.1:9880",
            "api_key": "k", "model": ["cosyvoice-v2"], "is_active": True, "priority": 50,
        })
        cosy_sent: list[httpx.Request] = []

        def cosy_handler(request: httpx.Request) -> httpx.Response:
            cosy_sent.append(request)
            return httpx.Response(200, json={"audio": base64.b64encode(b"ok").decode("ascii")})

        vendor_errors._vendor_client = httpx.AsyncClient(transport=httpx.MockTransport(cosy_handler))
        logged.clear()
        with engine.begin() as conn:
            cosy_path = run(tts.generate_tts(conn, {"text": "合成", "voice": "cosy-1",
                                                    "emotion": "happy", "speed": 1.3}))
        cosy_drops = {meta["field"]: meta["reason"] for kind, action, meta in logged
                      if action == "voice-contract-dropped"}
        check("⑫ ⭐⭐ CosyVoice 克隆路径：情绪**和**语速都进 ``dropped`` 且理由**点名克隆** ✗✗"
              "（此前两样都静默消失 ✓✗）",
              cosy_path.startswith("static/audio/") and set(cosy_drops) == {"emotion", "speed"}
              and all("克隆" in reason for reason in cosy_drops.values()), cosy_drops)
        check("⑫′ 请求是真发出去了的 ✓（证「报告了但活儿照干」✓ 不是整单挂掉 ✓）",
              len(cosy_sent) == 1 and str(cosy_sent[0].url).endswith("/inference_zero_shot"),
              [str(r.url) for r in cosy_sent])
    finally:
        tts.log_task_payload, tts.log_task_warn, tts.log_task_error = original


def main() -> int:
    case_resolve_emotion()
    case_capability_report()
    case_capability_speed()
    case_generate_tts_wiring()
    failures = [(name, detail) for name, passed, detail in _RESULTS if not passed]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    # ⚠️ Windows 中文控制台是 **GBK**（py3.14 及以前不会自动 UTF-8 ✗）⇒ 不加这行，裸跑第一条 `print` 就 `UnicodeEncodeError` 崩 ✗（守口径见 `check_cli_encoding.py` ✓）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
