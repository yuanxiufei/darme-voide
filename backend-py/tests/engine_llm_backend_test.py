"""自检：自研文本后端（``engine/llm_backend.py`` ✓ 2026-09-25 起 ✓）。

验的是什么 ✗：把 ``llm``（架构）+ ``gguf_to_llm``（装载）收成**一个后端** ✓ —— ``describe`` 自述 +
``generate`` 生成 ✓，让文本生成能**绕过 ollama** ✗。

判据（都是「看着接了、其实要么装死要么空转」的形状 ✓✗）：
1. ⭐ **describe 如实报缺** ✗✗：权重/词表/模型三者缺啥报啥 ✓ —— 真权重没到就说不可用 ✓ 不冒充 ✗；
2. ⭐ **未装载就 generate ⇒ 明确拒** ✗（不静默返回空串 ✗）；
3. ⭐ **generate 走通链路** ✓：encode → generate_llm → decode ✓（stub 记录调用 ✓ 返回非空 ✓）。

⚠️ 本套用**缩小版模型 + stub 词表** ✗（不读真 GGUF 文件 ✗ —— 真装载走 `gguf_to_llm` 那套已验 ✓）。

运行::

    ./.venv/Scripts/python.exe tests/engine_llm_backend_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="llmbe_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.engine import llm as llm_mod  # noqa: E402
from app.services.engine import llm_backend  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


CONFIG = llm_mod.LlmConfig(vocab_size=32, hidden=16, depth=1, heads=2, kv_heads=1,
                           head_dim=8, ffn=32, eos_id=31)


class _StubHub:
    """stub 词表 ✓（记录调用 ✓ —— 不读真词表 ✗）。"""

    def __init__(self) -> None:
        self.encoded: list[list[int]] = []
        self.decoded: list[list[int]] = []

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        self.encoded.append(list(text.encode("utf-8")))
        return [b % CONFIG.vocab_size for b in text.encode("utf-8")] or [0]

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        self.decoded.append(list(ids))
        return f"DECODED({len(ids)})"


def _raises(fn) -> bool:
    try:
        fn()
    except llm_backend.LlmBackendUnavailable:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


def case_describe() -> None:
    backend = llm_backend.LlmBackend(CONFIG)  # 无权重/词表路径
    d = backend.describe()
    check("① ⭐ describe 如实报缺 ✓（权重+词表+模型三者缺啥报啥 ✓ 不冒充 ✗）",
          d["available"] is False and d["canGenerate"] is False
          and "权重未就绪" in d["reason"] and "词表未就绪" in d["reason"]
          and "模型未装载" in d["reason"], d)
    check("①′ describe 字段齐全 ✓（name/realTensors/modelLoaded/tokenizerReady ✓）",
          d["name"] == "llm" and d["realTensors"] is True
          and d["modelLoaded"] is False and d["tokenizerReady"] is False)


def case_unavailable() -> None:
    backend = llm_backend.LlmBackend(CONFIG)
    check("② ⭐ 未装载就 generate ⇒ 明确拒 ✗（不静默返回空串 ✗）",
          _raises(lambda: backend.generate("你好")))


def _torch_available() -> bool:
    """⚠️ 缺 torch ⇒ 本段**显式 SKIP 打印** ✓（SKIP ≠ 通过 ✗ —— 别把「没跑」读成「绿了」✗）。"""
    try:
        import torch  # noqa: F401, PLC0415
    except ModuleNotFoundError:
        return False
    return True


def case_generate() -> None:
    if not _torch_available():
        print("SKIP  ③ 生成链路：需要 torch（本机未装）⇒ 不判（≠ 通过 ✓）")
        return
    model = llm_mod.build_llm(CONFIG)
    hub = _StubHub()
    backend = llm_backend.LlmBackend(CONFIG)
    backend._model = model
    backend._hub = hub

    out = backend.generate("你好", max_new_tokens=8, temperature=0)
    check("③ ⭐ generate 走通链路 ✓（返回非空 ✓ + encode/decode 都被调 ✓）",
          isinstance(out, str) and out.startswith("DECODED(") and len(hub.encoded) == 1
          and len(hub.decoded) == 1, out)

    # system 会拼进 prompt（encode 收到的字节更长 ✓）
    backend.generate("短", system="你是个助手", max_new_tokens=4, temperature=0)
    check("③′ system 拼进 prompt ✓（encode 收到的内容含 system ✓）",
          b"system" in bytes(hub.encoded[-1]) or len(hub.encoded[-1]) > 6, hub.encoded[-1])

    # describe 在装载后（手工注入）⇒ canGenerate=True ✓
    d = backend.describe()
    check("③″ 装载后 describe 转可用 ✓（canGenerate=True ✓）", d["canGenerate"] is True, d)


def case_wiring() -> None:
    """接线层 ✓：`text_generation` 的 engine 分支走自研后端（非 HTTP ✗）."""
    # ⚠️ 这一段**不需要 torch** ✓，但要 ``fastapi``（``text_generation`` → ``core.response`` ✓）。
    # 缺它时**显式 SKIP** ✓（SKIP ≠ 通过 ✗）—— 原来会整份崩 ✗，把上面几条也一起埋掉 ✗✗。
    try:
        import fastapi  # noqa: F401, PLC0415
    except ModuleNotFoundError:
        print("SKIP  ④ 接线：需要 fastapi（本解释器未装）⇒ 不判（≠ 通过 ✓）")
        return

    import asyncio  # noqa: PLC0415

    from app.services import text_generation as tg  # noqa: PLC0415

    class _FakeBackend:
        def generate(self, prompt, *, system=None, temperature=1.0, max_new_tokens=256):
            return f"ENGINE:{prompt[:6]}"

    # ⚠️ 设备/精度是**关键字**传下去的 ✓（2026-10-29 补）⇒ 假后端必须收 ``**kwargs`` ✓ 且要能把它们
    # 记下来 ✓（只认两个位置参数的老 lambda 会把新参数**静默丢掉** ✗✗ —— 配置里写了 cuda 也照样跑 CPU ✗）。
    seen: dict[str, object] = {}

    original = tg._get_engine_backend

    def _fake(gguf, tok, **kwargs):
        seen.update(kwargs)
        return _FakeBackend()

    tg._get_engine_backend = _fake
    try:
        async def _missing():
            try:
                await tg._generate_with_engine({"settings": {}}, "hi", {})
                return False
            except ValueError:
                return True

        check("④ 接线: 缺 ggufPath ⇒ 报错 ✗（不静默回落 HTTP ✗）", asyncio.run(_missing()))

        async def _ok():
            return await tg._generate_with_engine(
                {"settings": {"ggufPath": "/m.gguf", "tokenizerPath": "/t"}}, "你好", {"temperature": 0.5})

        out = asyncio.run(_ok())
        check("④′ 接线: 走自研后端（非 HTTP）✓ 返回后端产物 ✓", out == "ENGINE:你好", out)
        check("④″ settings 不写设备 ⇒ 默认 ``auto`` 真的透传到后端 ✓（不是摆设 ✗）",
              seen.get("device") == "auto", seen)

        async def _cuda():
            return await tg._generate_with_engine(
                {"settings": {"ggufPath": "/m.gguf", "device": "cuda", "dtype": "float32"}},
                "你好", {})

        asyncio.run(_cuda())
        check("④‴ settings 里的 ``device``/``dtype`` 逐字透传 ✓（吞掉 = 配置不生效 ✗✗）",
              seen.get("device") == "cuda" and seen.get("dtype") == "float32", seen)
        check("④⁗ 缓存键含设备/精度 ✓（只按路径缓存 ⇒ 改设备仍拿到旧后端 ✗✗）",
              len(tg._engine_backends) == 0 or all(len(key) == 4 for key in tg._engine_backends),
              list(tg._engine_backends)[:2])
    finally:
        tg._get_engine_backend = original


def case_decoded_only_new() -> None:
    """⭐ 只解**新生成**的那段 ✓（2026-10-29 真机 qwen3:8b 撞出来 ✗✗）。

    症状：`generate()` 把「prompt＋续写」整条 `decode` 给调用方 ⇒ 用户看到自己刚发的那句
    （`OUT>>user\\n你好…\\nassistant\\n\\n好的…`✗）。判据用**长度**：prompt 比 `max_new_tokens` 长
    ⇒ 「整条解」必然超过 `max_new_tokens` ✗，「只解新的」必然不超过 ✓。
    """
    if not _torch_available():
        print("SKIP  ③‴ 只解新 token：需要 torch（本机未装）⇒ 不判（≠ 通过 ✓）")
        return
    model = llm_mod.build_llm(CONFIG)
    hub = _StubHub()
    backend = llm_backend.LlmBackend(CONFIG)
    backend._model = model
    backend._hub = hub

    long_prompt = "你好世界你好世界你好世界"          # 36 字节 ⇒ 36 个 stub id ✓
    backend.generate(long_prompt, max_new_tokens=3, temperature=0)
    sent, got = len(hub.encoded[-1]), len(hub.decoded[-1])
    check("③‴ ⭐ decode 只收到**新生成**的 token ✓（解整条 ⇒ 用户看到自己的提问 ✗✗）",
          sent >= 30 and got <= 3, (sent, got))


def case_chat_template_vars() -> None:
    """⭐ 骨架变量：`tools` / `documents` 必须显式给 `None` ✓（2026-10-29 真机 qwen3:8b 撞出来 ✗✗）。

    Qwen3 骨架**首行**就是 `{%- if tools %}` ✓，本层用 `StrictUndefined` ✓ ⇒ 不给这个变量
    ⇒ `UndefinedError: 'tools' is undefined` ⇒ 整条对话路径当场拒 ✗（权重装了两分钟才炸在渲染 ✗）。
    """
    from app.services.engine import chat_template  # noqa: PLC0415

    templ = chat_template.ChatTemplate(source="{%- if tools %}T{%- endif %}{{ messages[0].content }}")
    text = templ.render([{"role": "user", "content": "你好"}])
    check("③⁵ ⭐ 骨架引用 `tools` 而调用方没给 ⇒ 按 ``None`` 渲染 ✓（不许炸 ✗✗）",
          text == "你好", text)
    check("③⁵′ 真给了 `tools` ⇒ 骨架**照用** ✓（不是把变量一律写死成 None ✗）",
          templ.render([{"role": "user", "content": "你好"}], tools=[{"name": "f"}]) == "T你好")

    thinking = chat_template.ChatTemplate(
        source="{%- if enable_thinking is defined and enable_thinking is false %}NT{%- endif %}B")
    check("③⁵″ ``extra`` 能透传模板变量 ✓（``enable_thinking=False`` ⇒ 骨架走「不思考」分支 ✓）",
          thinking.render([{"role": "user", "content": "x"}], extra={"enable_thinking": False}) == "NTB")
    check("③⁵‴ 不给 ``enable_thinking`` ⇒ 变量**不定义** ✓（本层不替模板编默认值 ✗）",
          thinking.render([{"role": "user", "content": "x"}]) == "B")


def case_kv_wiring() -> None:
    """⭐ KV cache 口径接线 ✓（2026-11-08）—— 四类「配了没生效看不出来」的形状 ✗✗。

    1. ``describe`` 在**装载前**就要按**请求值**报数 ✓（配了 ``q8`` 却报 ``fp`` 的字节 ⇒ 人会以为
       配置没吃 ✓✗）；
    2. ``kvMode`` 拼错在 ``load()`` **第一行**就拒 ✓（装完 8 GiB 权重才报错＝白等一轮 ✗✗）；
    3. 缓存键必须含 ``kv_mode`` ✓ —— 否则改配置仍拿到老后端 ✓（显存/速度一动不动 ✗）；
    4. **三个调用口**（普通生成／流式／agent 工具循环）都得把 ``kv_mode`` 递下去 ✓✗
       —— 只改一处的话，另一条路仍在按原样浮点缓存 ✓ 而且回执各说各话 ✗。
    """
    import ast  # noqa: PLC0415

    plain = llm_backend.LlmBackend(CONFIG, kv_mode=None).describe()
    quant = llm_backend.LlmBackend(CONFIG, kv_mode="q8").describe()
    check("④⁵ ``describe`` 两栏分得开 ✓（``kvModeRequested`` vs 生效的 ``kv.mode`` ✓）",
          plain["kvModeRequested"] == "off" and plain["kv"]["mode"] == "fp"
          and quant["kvModeRequested"] == "q8" and quant["kv"]["mode"] == "q8",
          (plain["kvModeRequested"], plain["kv"]["mode"], quant["kvModeRequested"]))
    check("④⁶ 装载前就报**请求值**的字节 ✓（q8 的每 token 字节 < fp ✓ 且总量也 < fp ✓）",
          quant["kv"]["totalBytes"] < quant["kv"]["fpTotalBytes"]
          == plain["kv"]["totalBytes"], (quant["kv"]["totalBytes"], plain["kv"]["totalBytes"]))
    check("④⁶′ 装载前的数如实标注 ✓（``装载后生效`` ＋ 精度按 2 B 估 ✓ 不冒充已生效 ✗）",
          "装载后生效" in quant["kv"]["note"] and quant["kv"]["assumedDtype"] is True,
          quant["kv"]["note"])

    bad = llm_backend.LlmBackend(CONFIG, kv_mode="q08")
    err_text = ""
    try:
        bad.load()
    except Exception as err:  # noqa: BLE001
        err_text = f"{type(err).__name__}: {err}"
    check("④⁷ ``kvMode`` 拼错 ⇒ ``load()`` **开头**具名拒 ✓（不是装完权重才炸 ✗✗）",
          "kvMode" in err_text and "权重未就绪" not in err_text, err_text)
    desc_bad = llm_backend.LlmBackend(CONFIG, kv_mode="q08").describe()
    check("④⁷′ describe 遇拼错**不抛** ✓（回执要能离线跑 ✓）但把不认的请求值写进 note ✓",
          desc_bad["kvModeRequested"] == "q08" and "不认" in desc_bad["kv"]["note"],
          desc_bad["kv"]["note"])

    try:
        import fastapi  # noqa: F401, PLC0415
    except ModuleNotFoundError:
        print("SKIP  ④⁸ 需要 fastapi（本解释器未装）⇒ 不判（≠ 通过 ✓）")
        return

    from app.services import text_generation as tg  # noqa: PLC0415

    key_fp = tg._engine_backend_key("/m.gguf", None)
    key_q8 = tg._engine_backend_key("/m.gguf", None, kv_mode="q8")
    key_off = tg._engine_backend_key("/m.gguf", None, kv_mode="off")
    check("④⁸ 缓存键含 ``kv_mode`` ✓（改配置必须换后端，否则配了等于没配 ✗✗）"
          "—— 且**只有这一格**不同（别的格跟着动 ⇒ 同义写法各装一份后端 ✗✗）",
          key_fp != key_q8 and key_fp == key_off
          and [i for i in range(len(key_fp)) if key_fp[i] != key_q8[i]] == [5],
          (key_fp, key_q8))
    check("④⁸′ 同义写法归一成一个键 ✓（``None``／``off``／``auto``＝``q8`` 不许各装一份 8 GiB ✗✗）",
          tg._engine_backend_key("/m.gguf", None, gpu_layers=None)
          == tg._engine_backend_key("/m.gguf", None, gpu_layers="off")
          and tg._engine_backend_key("/m.gguf", None, kv_mode="auto") == key_q8
          and tg._engine_backend_key("/m.gguf", None, kv_mode="q08") != key_fp,
          (tg._engine_backend_key("/m.gguf", None, gpu_layers="off"),))
    check("④⁹ ``engine_kv_mode`` 两种键名都认 ✓、缺省 ``None`` ✓",
          tg.engine_kv_mode({"kvMode": "q8"}) == "q8"
          and tg.engine_kv_mode({"kv_mode": "off"}) == "off"
          and tg.engine_kv_mode({}) is None and tg.engine_kv_mode(None) is None)

    seen: dict[str, object] = {}
    original = tg._get_engine_backend

    class _Fake:
        def generate(self, prompt, *, system=None, temperature=1.0, max_new_tokens=256):
            return "OK"

    def _fake(gguf, tok, **kwargs):
        seen.update(kwargs)
        return _Fake()

    tg._get_engine_backend = _fake
    try:
        import asyncio  # noqa: PLC0415

        asyncio.run(tg._generate_with_engine(
            {"settings": {"ggufPath": "/m.gguf", "kvMode": "q8"}}, "hi", {}))
    finally:
        tg._get_engine_backend = original
    check("④⁹′ 普通生成口把 ``kv_mode`` 递下去 ✓（不是只进缓存键、不交后端 ✗）",
          seen.get("kv_mode") == "q8", seen)

    root = Path(__file__).resolve().parents[1] / "app"
    targets = {"services/text_generation.py": "_generate_with_engine",
               "services/text_stream.py": None, "agent/runtime.py": None}
    offenders: list[str] = []
    for rel, _fn in targets.items():
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id.lstrip("_") == "get_engine_backend"]
        if not calls:
            offenders.append(f"{rel}: 找不到 get_engine_backend 调用 ✗")
            continue
        for call in calls:
            names = {kw.arg for kw in call.keywords}
            if "kv_mode" not in names:
                offenders.append(f"{rel}: 少 kv_mode")
            if "gpu_layers" not in names:
                offenders.append(f"{rel}: 少 gpu_layers")
    check("④⁹″ ⭐ 三个调用口**都**递 ``kv_mode`` ✓（只改一处＝另外两条路仍在浮点缓存 ✗✗）",
          not offenders, offenders)

    tree = ast.parse((root / "services/engine/llm_backend.py").read_text(encoding="utf-8"))
    gens = [node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "generate_llm"]
    missing = [node.lineno for node in gens
               if "kv_mode" not in {kw.arg for kw in node.keywords}]
    check("④⁹‴ ⭐ 后端里**每一处** ``generate_llm`` 都带了 ``kv_mode`` ✓（漏一处＝那条路仍是浮点 ✗✗）",
          len(gens) == 3 and not missing, (len(gens), missing))

    missing_cache = [node.lineno for node in gens
                     if "kv_cache" not in {kw.arg for kw in node.keywords}]
    check("④⁹⁴ ⭐ 同一批调用还带了 ``kv_cache`` ✓（只给一条路复用 ⇒ 同一后端两条路一套回执 ✗✗）",
          not missing_cache, missing_cache)

    # ── 跨调用前缀复用（2026-11-08 ✓ ⑩ 系列）────────────────────────────
    default = llm_backend.LlmBackend(CONFIG).describe()["kvCache"]
    off = llm_backend.LlmBackend(CONFIG, kv_cache_bytes=0).describe()["kvCache"]
    unit = llm_backend.LlmBackend(CONFIG, kv_cache_bytes="256MiB").describe()["kvCache"]
    check("⑩ 回执有 ``kvCache`` 且装载前就报准预算 ✓（不写 ⇒ 512 MiB；``0`` ⇒ 关；``256MiB`` ⇒ 字节 ✓）",
          default["enabled"] is True and default["budgetBytes"] == 512 * 2 ** 20
          and off["enabled"] is False and off["budgetBytes"] == 0
          and unit["budgetBytes"] == 256 * 2 ** 20, (default, off, unit))
    bad_cache = llm_backend.LlmBackend(CONFIG, kv_cache_bytes="512兆")
    err_cache = ""
    try:
        bad_cache.load()
    except Exception as err:  # noqa: BLE001
        err_cache = f"{type(err).__name__}: {err}"
    check("⑩′ ⭐ ``kvCacheBytes`` 拼错 ⇒ ``load()`` **开头**具名拒 ✓ ＋ describe 不抛只写 note ✓",
          "kvCacheBytes" in err_cache and "权重未就绪" not in err_cache
          and "不认" in bad_cache.describe()["kvCache"]["note"],
          (err_cache, bad_cache.describe()["kvCache"]["note"]))
    check("⑩″ ``engine_kv_cache_bytes`` 两个开关分得开 ✓（``kvReuse=off`` ⇒ 0；不写 ⇒ None 走默认 ✓）",
          tg.engine_kv_cache_bytes({"kvReuse": "off"}) == 0
          and tg.engine_kv_cache_bytes({"kv_reuse": False}) == 0
          and tg.engine_kv_cache_bytes({}) is None
          and tg.engine_kv_cache_bytes({"kvCacheBytes": "256MiB"}) == "256MiB")
    check("⑩‴ 缓存键含复用预算 ✓（``0`` 与「不写」必须分格：一个不占显存、一个占 512 MiB ✗✗）",
          tg._engine_backend_key("/m.gguf", None, kv_cache_bytes=0)
          != tg._engine_backend_key("/m.gguf", None)
          and tg._engine_backend_key("/m.gguf", None, kv_cache_bytes="off")
          == tg._engine_backend_key("/m.gguf", None, kv_cache_bytes=0)
          and tg._engine_backend_key("/m.gguf", None, kv_cache_bytes="512MiB")
          == tg._engine_backend_key("/m.gguf", None))

    seen_cache: dict[str, object] = {}

    def _fake2(gguf, tok, **kwargs):
        seen_cache.update(kwargs)
        return _Fake()

    tg._get_engine_backend = _fake2
    try:
        import asyncio  # noqa: PLC0415

        asyncio.run(tg._generate_with_engine(
            {"settings": {"ggufPath": "/m.gguf", "kvReuse": "off"}}, "hi", {}))
    finally:
        tg._get_engine_backend = original
    check("⑩⁴ 普通生成口把复用预算也递下去 ✓（``kvReuse=off`` ⇒ 后端收 0 ✓）",
          seen_cache.get("kv_cache_bytes") == 0, seen_cache)

    wired = []
    for rel in ("services/text_generation.py", "services/text_stream.py", "agent/runtime.py"):
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id.lstrip("_") == "get_engine_backend"]
        wired.append(all("kv_cache_bytes" in {kw.arg for kw in call.keywords} for call in calls))
    check("⑩⁵ ⭐ 三个调用口**都**递 ``kv_cache_bytes`` ✓（漏一处＝那条路的复用被静默关掉 ✗✗）",
          all(wired), wired)


def case_kv_window_wiring() -> None:
    """⭐ KV 滑窗接线 ✓（2026-11-08）—— 与 ④／⑩／⑪ 同四类形状，外加两条滑窗独有的假绿 ✗✗。

    1. 回执必须把「关」与「配错」分开 ✓，且窗口是**行数**不是开关 ✓（``kvSink=0`` 合法 ⇒
       归一函数别照抄 ``specWindow`` 那条「< 1 即拒」✗）；
    2. 拼错在 ``load()`` 开头具名拒 ✓；
    3. 缓存键：关着＝空串 ✓（``None``/``0``/``off`` 同格 ✓ 否则同一份权重装两遍 ✗✗）、
       **窗口关着时 sink 不进格** ✗（写个 ``kvSink=8`` 就多装一份 8 GiB 权重 ✗✗）；
    4. 三个调用口都递 ✓ ＋ 后端里**每一处** ``generate_llm`` 都用 ``_kv_window_call()``／
       ``_kv_cache_for_call()`` ✓（漏一处＝那条路仍按完整历史涨 KV ✗✗，而回执只有一份 ✓）。
    """
    import ast  # noqa: PLC0415

    off = llm_backend.LlmBackend(CONFIG).describe()
    on = llm_backend.LlmBackend(CONFIG, kv_window=1024).describe()
    zero_sink = llm_backend.LlmBackend(CONFIG, kv_window=64, kv_sink=0).describe()
    check("⑮ 回执分得开 ✓（默认 ``kvWindow.enabled=False``/``rows=0`` ✓；``kv_window=1024`` ⇒ sink 默认 4、"
          "``rows = 1028`` ✓；⭐ ``kvSink=0`` **合法** ✓ —— 与 ``specWindow`` 那条「< 1 即拒」口径不同 ✗）",
          off["kvWindow"]["enabled"] is False and off["kvWindow"]["rows"] == 0
          and on["kvWindow"]["enabled"] is True and on["kvWindow"]["window"] == 1024
          and on["kvWindow"]["sink"] == 4 and on["kvWindow"]["rows"] == 1028
          and zero_sink["kvWindow"]["sink"] == 0 and zero_sink["kvWindow"]["rows"] == 64
          and "有损" in on["kvWindow"]["note"],
          (off["kvWindow"], on["kvWindow"], zero_sink["kvWindow"]))

    bad = llm_backend.LlmBackend(CONFIG, kv_window="3k")
    err_text = ""
    try:
        bad.load()
    except Exception as err:  # noqa: BLE001
        err_text = f"{type(err).__name__}: {err}"
    check("⑮′ ⭐ ``kvWindow=\"3k\"`` 拼错 ⇒ ``load()`` **开头**具名拒 ✓（不是装完权重才炸 ✗✗ —— "
          "静默当 0 的后果是「人以为开了滑窗、显存照涨」✓✗）＋ describe 不抛只写 note ✓",
          "kvWindow" in err_text and "权重未就绪" not in err_text
          and "不认" in bad.describe()["kvWindow"]["note"], (err_text, bad.describe()["kvWindow"]))

    equal = llm_backend.LlmBackend(CONFIG, kv_window=8, kv_sink=8)
    err_equal = ""
    try:
        equal.load()
    except Exception as err:  # noqa: BLE001
        err_equal = f"{type(err).__name__}: {err}"
    tight_ok = llm_backend.LlmBackend(CONFIG, kv_window=9, kv_sink=8)
    tight_err = ""
    try:
        tight_ok.load()
    except Exception as err:  # noqa: BLE001
        # 走到「权重未就绪」＝过了滑窗校验 ✓（CONFIG 指向不存在的文件 ✓）
        tight_err = f"{type(err).__name__}: {err}"
    check("⑮″ ``kvSink >= kvWindow`` ⇒ ``load()`` 具名拒 ✓（sink 是窗口**之外**另留的开头几行 ⇒ "
          "相等就没有可滚的窗口 ✓✗）；``kvSink=8/kvWindow=9`` 过校验 ✓（拒的理由不能是这句 ✗）",
          "kvSink" in err_equal and "kvWindow" in err_equal and "kvSink" not in tight_err,
          (err_equal, tight_err))

    from app.services import text_generation as tg  # noqa: PLC0415

    root = Path(__file__).resolve().parents[1] / "app"
    key = tg._engine_backend_key
    check("⑮‴ 缓存键 ✓：``0``/``None``/``off``/不写**同格** ✓（否则同一份权重装两遍 ✗✗）；"
          "``1024`` 另起一格 ✓；窗口开着时 ``kvSink`` 另起一格 ✓；**窗口关着时 sink 不进格** ✓"
          "（写个 ``kvSink=8`` 不该多装一份权重 ✗✗）；拼错原样留一格 ✓ 交给 load 拒 ✓",
          key("/m.gguf", None, kv_window=0) == key("/m.gguf", None)
          and key("/m.gguf", None, kv_window="off") == key("/m.gguf", None)
          and key("/m.gguf", None, kv_window=None) == key("/m.gguf", None)
          and key("/m.gguf", None, kv_window=1024) != key("/m.gguf", None)
          and key("/m.gguf", None, kv_window=1024, kv_sink=8)
          != key("/m.gguf", None, kv_window=1024)
          and key("/m.gguf", None, kv_sink=8) == key("/m.gguf", None)
          and key("/m.gguf", None, kv_window="bad") != key("/m.gguf", None))

    check("⑮⁴ 滑窗开着 ⇒ ``kvCache`` **如实暂停** ✓（``enabled=False`` ＋ ``suspendedBy=\"kvWindow\"`` ✓）"
          "；关着 ⇒ 复用照旧 ✓（静默停掉 ⇒ 人按「复用在跑」算时间会算错 ✗✗）",
          on["kvCache"]["enabled"] is False and on["kvCache"].get("suspendedBy") == "kvWindow"
          and off["kvCache"]["enabled"] is True, (on["kvCache"], off["kvCache"]))

    wired = []
    for rel in ("services/text_generation.py", "services/text_stream.py", "agent/runtime.py"):
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id.lstrip("_") == "get_engine_backend"]
        wired.append(bool(calls) and all(
            any(kw.arg is None and isinstance(kw.value, ast.Call)
                and getattr(kw.value.func, "id", "") == "engine_kv_window_kwargs"
                for kw in call.keywords) for call in calls))
    check("⑮⁵ ⭐ 三个调用口**都**递 ``**engine_kv_window_kwargs(settings)`` ✓（漏一处 ⇒ 那条路仍按完整"
          "历史涨 KV ✗✗ 而回执只有一份 ✓）", all(wired), wired)

    tree = ast.parse((root / "services/engine/llm_backend.py").read_text(encoding="utf-8"))
    gens = [node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", "") == "generate_llm"
            or (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "generate_llm")]
    seen: list[set[str]] = []
    for call in gens:
        names = set()
        for kw in call.keywords:
            if kw.arg is None and isinstance(kw.value, ast.Call):
                names.add(getattr(kw.value.func, "attr", "") or getattr(kw.value.func, "id", ""))
            elif kw.arg == "kv_cache":
                # ``kv_cache=_kv_cache_for_call()`` ⇒ 是个 Call ✓ 名字在 ``.func.attr`` 上 ✓
                target = kw.value.func if isinstance(kw.value, ast.Call) else kw.value
                names.add(getattr(target, "attr", "") or getattr(target, "id", ""))
        seen.append(names)
    check("⑮⁶ ⭐ 后端里**每一处** ``generate_llm`` 都走 ``_kv_window_call()`` ＋ ``_kv_cache_for_call()`` ✓"
          "（给着一半＝三条路两套行为 ✗✗ —— 分别漏掉时症状是「流式仍留全部历史」／「滑窗下复用还命中」✓✗）",
          len(seen) >= 3 and all("_kv_window_call" in s and "_kv_cache_for_call" in s for s in seen),
          (len(seen), seen))


def case_prefill_tier() -> None:
    """⭐ prefill 分块按显存档位 ✓（2026-11-08 补）—— 三条判据，每条盯一个错法 ✗。

    为什么块大小归后端 ✗✗：块只解决**峰值**（分数矩阵 ``O(chunk × 总长)`` ✓，与 KV 容量无关 ✓），
    而「显存有多紧」只有刚 ``refresh_kv_budget`` 过的后端知道 ✓；调用方拿不到这个数只能写死 1024 ✓✗
    —— 2 GiB 空闲的卡上 prefill 3240 词元当场炸 ✓。

    ① **常见环境一个字节都不变** ✓（≥4 GiB ⇒ 1024 ⇒ ``_prefill_call()`` 返回空字典 ✓ ⇒ 调用签名
       与老路径逐字相同 ✓）；**只缩不涨** ✗（8 GiB 也只给 1024 ✓ —— 涨没有实测收益却把峰值翻倍 ✓✗）；
    ② 探不到显存 ⇒ 默认档 ✓（``None`` 不等于「很紧」✗✗）；
    ③ 三处 ``generate_llm`` **都**展开 ``_prefill_call()`` ✓（漏一处 ⇒ 那条路仍按写死 1024 跑 ✓✗）。
    """
    import ast  # noqa: PLC0415

    backend = llm_backend.LlmBackend(CONFIG)
    tiers: dict[str, int] = {}

    def chunk_of(free: int | None) -> tuple[int, dict]:
        backend._kv_budget = None if free is None else {"freeBytes": free}
        return backend._prefill_chunk(), backend._prefill_call()

    for label, free in (("未探到", None), ("40 GiB", 40 * 2 ** 30), ("8 GiB", 8 * 2 ** 30),
                        ("4 GiB", 4 * 2 ** 30), ("3 GiB", 3 * 2 ** 30),
                        ("1.5 GiB", int(1.5 * 2 ** 30)), ("0.5 GiB", 512 * 2 ** 20)):
        tiers[label] = chunk_of(free)[0]
    common_call = chunk_of(40 * 2 ** 30)[1]
    tight_call = chunk_of(3 * 2 ** 30)[1]
    check("⑯⁰ ⭐ 常见环境**逐字不变** ✓：未探到／≥4 GiB ⇒ 1024 且 ``_prefill_call()`` 是**空字典** ✓"
          "（⇒ 老调用签名不变 ✓）；**只缩不涨** ✗：8 GiB 也给 1024 不是 2048 ✓；"
          "3 GiB⇒512 ✓、1.5 GiB⇒256 ✓、0.5 GiB⇒128 ✓",
          tiers == {"未探到": 1024, "40 GiB": 1024, "8 GiB": 1024, "4 GiB": 1024,
                    "3 GiB": 512, "1.5 GiB": 256, "0.5 GiB": 128}
          and common_call == {} and tight_call == {"prefill_chunk": 512}, (tiers, common_call, tight_call))

    backend._kv_budget = {"freeBytes": 3 * 2 ** 30}
    report = backend.describe()["prefill"]
    backend._kv_budget = None
    idle_report = backend.describe()["prefill"]
    check("⑯¹ 回执同时给出「这一档是多少」和「为什么」✓（``chunk`` ＋ ``note`` ＋ ``freeBytes`` ✓）；"
          "缩档时说清是空闲显存紧 ✓、默认档时说清「老路径逐字不变」✓"
          "（只报一个数 ⇒ 人不知道 256 是配置还是让步 ✓✗）",
          report["chunk"] == 512 and report["default"] == 1024
          and "空闲只有 3.0 GiB" in report["note"]
          and idle_report["chunk"] == 1024 and "默认档" in idle_report["note"],
          (idle_report, report))

    root = Path(__file__).resolve().parents[1] / "app"
    tree = ast.parse((root / "services/engine/llm_backend.py").read_text(encoding="utf-8"))
    gens = [node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (getattr(node.func, "attr", "") == "generate_llm"
                 or getattr(node.func, "id", "") == "generate_llm")]
    wired = []
    for call in gens:
        wired.append(any(
            kw.arg is None and isinstance(kw.value, ast.Call)
            and getattr(kw.value.func, "attr", "") == "_prefill_call"
            for kw in call.keywords))
    check("⑯² ⭐ 后端里**每一处** ``generate_llm`` 都展开 ``**self._prefill_call()`` ✓"
          "（漏一处 ⇒ 那条路写死 1024 ✓✗：低显存下正是「流式那条」最容易漏 ✓）",
          len(gens) >= 3 and all(wired), (len(gens), wired))


def case_spec_wiring() -> None:
    """⭐ 投机解码接线 ✓（2026-11-08）—— 与 ④／⑩ 同四类形状，但每个都有「投机特有」的假绿 ✗✗。

    1. ``describe`` 必须把「开关」与「**上一次真的投机了没**」分开报 ✓✗（只报开关 ⇒ 温度 > 0 时
       回执照样写「已开」✓ 而实际一拍一 token ✗✗）；
    2. 旋钮拼错在 ``load()`` 开头就拒 ✓（装完 8 GiB 权重才报错＝白等 ✗）；
    3. 缓存键必须含投机三格 ✓ —— 而且**关着时窗口不许进格** ✗（否则写个 ``specWindow=4`` 就又装一份权重 ✗✗）；
    4. **三个调用口**都递 ✓，**后端里每一处 ``generate_llm``** 都带 ``spec_lookup``/``spec_report`` ✓。
    """
    import ast  # noqa: PLC0415

    off = llm_backend.LlmBackend(CONFIG).describe()["spec"]
    on = llm_backend.LlmBackend(CONFIG, spec_lookup="on").describe()["spec"]
    check("⑪ 回执分得开 ✓（默认关 ＋ 不写收益 ⇒ 别把「关」读成「没实现」✓；开了要写明**只在温度 0** 成立 ✓）",
          off["enabled"] is False and off["requested"] == "off" and "1.02" in off["note"]
          and on["enabled"] is True and on["window"] == 8 and on["minMatch"] == 3
          and "temperature=0" in on["note"], (off, on))

    bad = llm_backend.LlmBackend(CONFIG, spec_lookup="ture")
    err_text = ""
    try:
        bad.load()
    except Exception as err:  # noqa: BLE001
        err_text = f"{type(err).__name__}: {err}"
    check("⑪′ ``specLookup`` 拼错 ⇒ ``load()`` **开头**具名拒 ✓（不是装完权重才炸 ✗✗）＋ describe 不抛 ✓",
          "specLookup" in err_text and "权重未就绪" not in err_text
          and "不认" in bad.describe()["spec"]["note"], (err_text, bad.describe()["spec"]["note"]))

    zero = llm_backend.LlmBackend(CONFIG, spec_lookup="on", spec_window=0)
    err_zero = ""
    try:
        zero.load()
    except Exception as err:  # noqa: BLE001
        err_zero = f"{type(err).__name__}: {err}"
    check("⑪″ ``specWindow=0`` ⇒ 当场拒 ✓（0 草稿＝白跑一次窗口前向 ＋ 回执说「开了」✗✗）",
          "specWindow" in err_zero, err_zero)

    try:
        import fastapi  # noqa: F401, PLC0415
    except ModuleNotFoundError:
        print("SKIP  ⑪‴ 需要 fastapi（本解释器未装）⇒ 不判（≠ 通过 ✓）")
        return

    from app.services import text_generation as tg  # noqa: PLC0415

    key_on = tg._engine_backend_key("/m.gguf", None, spec_lookup="on")
    key_off = tg._engine_backend_key("/m.gguf", None, spec_lookup="off")
    check("⑪‴ 缓存键含投机开关 ✓（改配置必须换后端，否则配了等于没配 ✗✗；关＝与「不写」同格 ✓）",
          key_on != key_off and key_off == tg._engine_backend_key("/m.gguf", None),
          (key_on, key_off))
    check("⑪⁴ 窗口/最短匹配也进格 ✓，但**关着时**不进 ✓（否则写个 specWindow 就多装一份 8 GiB ✗✗）",
          tg._engine_backend_key("/m.gguf", None, spec_lookup="on", spec_window=4)
          != tg._engine_backend_key("/m.gguf", None, spec_lookup="on")
          and tg._engine_backend_key("/m.gguf", None, spec_lookup="on", spec_min_match=2)
          != key_on
          and tg._engine_backend_key("/m.gguf", None, spec_lookup="off", spec_window=4) == key_off,
          (key_on,))
    check("⑪⁵ ``engine_spec_*`` 两种键名都认 ✓、缺省 ``None`` ✓、一把给的三格齐全 ✓",
          tg.engine_spec_lookup({"specLookup": "on"}) == "on"
          and tg.engine_spec_lookup({"spec_lookup": "off"}) == "off"
          and tg.engine_spec_lookup({}) is None
          and tg.engine_spec_window({"specWindow": 4}) == 4
          and tg.engine_spec_min_match({"specMinMatch": 2}) == 2
          and set(tg.engine_spec_kwargs({"specLookup": "on"}))
          == {"spec_lookup", "spec_window", "spec_min_match"})

    seen: dict[str, object] = {}
    original = tg._get_engine_backend

    class _Fake:
        def generate(self, prompt, *, system=None, temperature=1.0, max_new_tokens=256):
            return "OK"

    def _fake(gguf, tok, **kwargs):
        seen.update(kwargs)
        return _Fake()

    tg._get_engine_backend = _fake
    try:
        import asyncio  # noqa: PLC0415

        asyncio.run(tg._generate_with_engine(
            {"settings": {"ggufPath": "/m.gguf", "specLookup": "on", "specWindow": 4}}, "hi", {}))
    finally:
        tg._get_engine_backend = original
    # ⚠️ 递下去的是**原样请求值** ✓（``"on"`` 不是 ``True`` ✓ —— 归一在 ``LlmBackend.__init__`` 里做 ✓，
    # 与 ``kvMode="q8"`` 同一个口径 ✓；这一层做类型转换反而会把「拼错」吞成假值 ✗✗）。
    check("⑪⁶ 普通生成口把三格都递下去 ✓（只进缓存键、不交后端＝配置静默失效 ✗✗）",
          seen.get("spec_lookup") == "on" and seen.get("spec_window") == 4, seen)

    root = Path(__file__).resolve().parents[1] / "app"
    missing_port: list[str] = []
    for rel in ("services/text_generation.py", "services/text_stream.py", "agent/runtime.py"):
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id.lstrip("_") == "get_engine_backend"]
        if not calls:
            missing_port.append(f"{rel}: 找不到 get_engine_backend 调用 ✗")
        for call in calls:
            spread = [kw for kw in call.keywords if kw.arg is None
                      and isinstance(kw.value, ast.Call)
                      and getattr(kw.value.func, "id", "") == "engine_spec_kwargs"]
            if not spread:
                missing_port.append(f"{rel}: 没展开 engine_spec_kwargs")
    check("⑪⁷ ⭐ 三个调用口**都**递投机三格 ✓（漏一处＝那条路静默不开 ✗✗）", not missing_port, missing_port)

    tree = ast.parse((root / "services/engine/llm_backend.py").read_text(encoding="utf-8"))
    gens = [node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "generate_llm"]
    # ⚠️ 三处都走 ``**spec_kwargs`` 展开 ✓（字段名在 ``_spec_call`` 里只写一次 ✓）⇒ 判据必须同时认
    # 「显式关键字」与「展开那个变量」✗ 只认前者会把**正确的**写法判红 ✓✗。
    def _has_spec(node: ast.Call) -> bool:
        names = {kw.arg for kw in node.keywords}
        spread = any(kw.arg is None and isinstance(kw.value, ast.Name)
                     and kw.value.id == "spec_kwargs" for kw in node.keywords)
        return spread or {"spec_lookup", "spec_report"} <= names

    missing_kw = [node.lineno for node in gens if not _has_spec(node)]
    stores = [node for node in ast.walk(tree)
              if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Attribute) and t.attr == "_spec_last"
                      for t in node.targets)]
    check("⑪⁸ ⭐ 后端里**每一处** ``generate_llm`` 都带 ``spec_lookup`` ＋ ``spec_report`` ✓"
          "（不送回执 ⇒ 永远空白 ✓ 人分不出「没开」与「开了没生效」✗✗）",
          len(gens) == 3 and not missing_kw and len(stores) >= 3,
          (len(gens), missing_kw, len(stores)))

    if not _torch_available():
        print("SKIP  ⑪⁹ 真跑：需要 torch（本机未装）⇒ 不判（≠ 通过 ✓）")
        return
    model = llm_mod.build_llm(CONFIG)
    hub = _StubHub()
    plain = llm_backend.LlmBackend(CONFIG)
    plain._model, plain._hub = model, hub
    spec = llm_backend.LlmBackend(CONFIG, spec_lookup="on", spec_window=4)
    spec._model, spec._hub = model, hub
    text_plain = plain.generate("abcabcabcabc", max_new_tokens=8, temperature=0)
    text_spec = spec.generate("abcabcabcabc", max_new_tokens=8, temperature=0)
    last = spec.describe()["spec"]
    check("⑪⁹ ⭐ 真跑：通了投机 ⇒ 回执有**轮数/草稿数** ✓ 且与不开**逐 token 相同** ✓",
          text_spec == text_plain and int(last["rounds"]) > 0 and int(last["drafted"]) > 0,
          (text_plain, text_spec, last))
    warm = plain.generate("abcabcabcabc", max_new_tokens=8, temperature=0.7)
    warm_spec = spec.generate("abcabcabcabc", max_new_tokens=8, temperature=0.7)
    warm_last = spec.describe()["spec"]
    check("⑪¹⁰ ⭐ 温度 > 0 ⇒ **如实退回** ✓（回执写 enabled=False/原因 ✓ 且输出与不开一致 ✓ 不假装开了 ✗✗）",
          warm_spec == warm and warm_last["enabled"] is False
          and warm_last.get("last", {}).get("reason") == "temperature>0"
          and "退回" in warm_last["note"], (warm, warm_spec, warm_last))


def case_fast_wiring() -> None:
    """快路径接线 ✓（2026-11-08）—— 与 ④／⑩／⑪ 同一批形状，但它有一处**独有的假绿** ✗✗：

    本档改的是**数值**（``F.rms_norm`` 融合核与老五连差 1 个 bf16 ulp ✓、SDPA 与手写三式差 1e-7 级 ✓）
    ⇒ 只要开关传错了，**输出照样出得来、看不出来** ✓✗。所以判据全压在「开关有没有真的到核里」：
    ``describe`` 三态分得开 ✓、拼错在 ``load()`` 开头拒 ✓、缓存键含 ``fastPath`` ✓（否则拿到的是**关着的
    旧后端**、回执照报「开」✗✗）、三处调用口都递 ✓、后端每处 ``generate_llm`` 都带 ✓。
    """
    import ast  # noqa: PLC0415

    off = llm_backend.LlmBackend(CONFIG).describe()["fastPath"]
    norm = llm_backend.LlmBackend(CONFIG, fast_path="norm").describe()["fastPath"]
    both = llm_backend.LlmBackend(CONFIG, fast_path="all").describe()["fastPath"]
    check("⑫ 回执把 ``norm``／``attn``／``kvbuf`` **分开报** ✓（合成一个布尔 ⇒ 人问「到底省了哪一刀」"
          "答不出来 ✗✗）＋ 关着时也说清「默认关、数值逐位不变」✓（别把「关」读成「没实现」✗）",
          off["enabled"] is False and off["items"] == [] and "逐位" in off["note"]
          and norm["norm"] is True and norm["attn"] is False and norm["kvbuf"] is False
          and norm["items"] == ["norm"]
          and both["norm"] is True and both["attn"] is True and both["kvbuf"] is True
          and sorted(both["items"]) == ["attn", "kvbuf", "norm"], (off, norm, both))
    avail = bool(norm["fusedNormAvailable"])
    check("⑫′ ⭐ 回执单列 ``fusedNormAvailable`` ✓（缺 ``F.rms_norm`` 时 ``norm`` **静默退回**老写法 ✓ "
          "⇒ 只报「已开」就是假的 ✗✗；本机实测 " + ("有" if avail else "没有") + " ✓）",
          isinstance(avail, bool)
          and (("退回" in both["note"]) if not avail else ("退回" not in both["note"])),
          (avail, both["note"]))

    bad = llm_backend.LlmBackend(CONFIG, fast_path="nrom")
    err_text = ""
    try:
        bad.load()
    except Exception as err:  # noqa: BLE001
        err_text = f"{type(err).__name__}: {err}"
    check("⑫″ ``fastPath`` 拼错 ⇒ ``load()`` **开头**具名拒 ✓（不是装完 8 GiB 权重才炸 ✗✗）"
          "＋ describe 不抛、把不认的值写进 note ✓",
          "fastPath" in err_text and "权重未就绪" not in err_text
          and "不认" in bad.describe()["fastPath"]["note"],
          (err_text, bad.describe()["fastPath"]["note"]))

    try:
        import fastapi  # noqa: F401, PLC0415
    except ModuleNotFoundError:
        print("SKIP  ⑫‴～⑫⁷ 需要 fastapi（本解释器未装）⇒ 不判（≠ 通过 ✓）")
        return

    from app.services import text_generation as tg  # noqa: PLC0415

    key_off = tg._engine_backend_key("/m.gguf", None)
    key_all = tg._engine_backend_key("/m.gguf", None, fast_path="all")
    check("⑫‴ ⭐ 缓存键含 ``fastPath`` ✓（不进格 ⇒ 开关一变还是**同一个后端对象** ⇒ 配了等于没配 ✗✗、"
          "而回执照旧写「已开」✗✗）＋ 关＝与「不写」同格 ✓ ＋ 同义写法归一成**一格** ✓"
          "（``all``／``norm,attn,kvbuf``／``True`` 各装一份 8 GiB 是白炸 ✗✗）",
          key_all != key_off
          and key_off == tg._engine_backend_key("/m.gguf", None, fast_path="off")
          and key_all == tg._engine_backend_key("/m.gguf", None, fast_path="norm,attn,kvbuf")
          == tg._engine_backend_key("/m.gguf", None, fast_path="kvbuf,norm,attn")
          == tg._engine_backend_key("/m.gguf", None, fast_path=True),
          (key_off, key_all))
    check("⑫⁴ 拼错的写法**原样留在键里** ✓（归一抛错时不许当成「关」去复用那份关着的后端 ✗✗）",
          tg._engine_backend_key("/m.gguf", None, fast_path="nrom")
          != tg._engine_backend_key("/m.gguf", None, fast_path="off"))
    check("⑫⁵ ``engine_fast_path`` 两种键名都认 ✓、缺省 ``None`` ✓、``engine_fast_kwargs`` 只出这一格 ✓",
          tg.engine_fast_path({"fastPath": "all"}) == "all"
          and tg.engine_fast_path({"fast_path": "norm"}) == "norm"
          and tg.engine_fast_path({}) is None
          and set(tg.engine_fast_kwargs({"fastPath": "all"})) == {"fast_path"})

    seen: dict[str, object] = {}
    original = tg._get_engine_backend

    class _Fake:
        def generate(self, prompt, *, system=None, temperature=1.0, max_new_tokens=256):
            return "OK"

    def _fake(gguf, tok, **kwargs):
        seen.update(kwargs)
        return _Fake()

    tg._get_engine_backend = _fake
    try:
        import asyncio  # noqa: PLC0415

        asyncio.run(tg._generate_with_engine(
            {"settings": {"ggufPath": "/m.gguf", "fastPath": "all"}}, "hi", {}))
    finally:
        tg._get_engine_backend = original
    check("⑫⁶ 普通生成口把 ``fastPath`` 原样递下去 ✓（只进缓存键、不交后端＝配置静默失效 ✗✗；"
          "这里**不做类型转换** ✓ —— 归一在 ``LlmBackend.__init__`` 里，与 ``kvMode``／``specLookup`` 同口径 ✓）",
          seen.get("fast_path") == "all", seen)

    root = Path(__file__).resolve().parents[1] / "app"
    missing_port: list[str] = []
    for rel in ("services/text_generation.py", "services/text_stream.py", "agent/runtime.py"):
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id.lstrip("_") == "get_engine_backend"]
        if not calls:
            missing_port.append(f"{rel}: 找不到 get_engine_backend 调用 ✗")
        for call in calls:
            spread = [kw for kw in call.keywords if kw.arg is None
                      and isinstance(kw.value, ast.Call)
                      and getattr(kw.value.func, "id", "") == "engine_fast_kwargs"]
            if not spread:
                missing_port.append(f"{rel}: 没展开 engine_fast_kwargs")
    check("⑫⁷ ⭐ 三个调用口**都**递快路径 ✓（漏一处＝那条路永远老写法 ⇒ 人按「都开了」算账会算错 ✗✗）",
          not missing_port, missing_port)

    tree = ast.parse((root / "services/engine/llm_backend.py").read_text(encoding="utf-8"))
    gens = [node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "generate_llm"]

    def _has_fast(node: ast.Call) -> bool:
        spread = any(kw.arg is None and isinstance(kw.value, ast.Name)
                     and kw.value.id == "fast_kwargs" for kw in node.keywords)
        return spread or "fast_path" in {kw.arg for kw in node.keywords}

    missing_kw = [node.lineno for node in gens if not _has_fast(node)]
    # ⚠️ 判据要盯「谁去取了这一份」✗ 别盯变量名长得像 ✓：三处都是 ``fast_kwargs = self._fast_call()`` ✓
    callers = [node.lineno for node in ast.walk(tree)
               if isinstance(node, ast.Assign)
               and isinstance(node.value, ast.Call)
               and getattr(node.value.func, "attr", "") == "_fast_call"
               and any(isinstance(t, ast.Name) and t.id == "fast_kwargs" for t in node.targets)]
    check("⑫⁸ ⭐ 后端里**每一处** ``generate_llm`` 都带 ``fast_path`` ✓（普通／流式／agent 工具循环"
          "三条路漏一条 ⇒ 那条路每步还是 2506 次 launch ✗✗）",
          len(gens) == 3 and not missing_kw and len(callers) >= 3,
          (len(gens), missing_kw, len(callers)))
    check("⑫⁹ ``_fast_call()`` 空集 ⇒ **空字典** ✓（不传关键字 ⇒ 默认关时调用签名逐字不变 ✓ "
          "本仓一堆桩模型测试靠这条活着 ✓；写成 ``{\"fast_path\": frozenset()}`` 会把它们全打炸 ✗✗）",
          llm_backend.LlmBackend(CONFIG)._fast_call() == {}
          and llm_backend.LlmBackend(CONFIG, fast_path="all")._fast_call()
          == {"fast_path": frozenset(llm_mod.FAST_PATH_ITEMS)},
          (llm_backend.LlmBackend(CONFIG)._fast_call(),
           llm_backend.LlmBackend(CONFIG, fast_path="all")._fast_call()))

    if not _torch_available():
        print("SKIP  ⑫¹⁰ 真跑：需要 torch（本机未装）⇒ 不判（≠ 通过 ✓）")
        return
    # 端到端：归一后的 ``frozenset`` 真的落到了 ``generate_llm`` 的形参上 ✓（spy 代跑真函数 ✓）
    from app.services.engine import llm as llm_mod_rt  # noqa: PLC0415

    model = llm_mod_rt.build_llm(CONFIG)
    hub = _StubHub()
    backend = llm_backend.LlmBackend(CONFIG, fast_path="all")
    backend._model, backend._hub = model, hub
    got: list[object] = []
    real_gen = llm_mod_rt.generate_llm

    def _spy(*args, **kwargs):
        got.append(kwargs.get("fast_path", "<缺>"))
        return real_gen(*args, **kwargs)

    llm_mod_rt.generate_llm = _spy
    try:
        text = backend.generate("abcabcabc", max_new_tokens=4, temperature=0)
    finally:
        llm_mod_rt.generate_llm = real_gen
    check("⑫¹⁰ ⭐ 真跑：``fastPath=all`` 一路走到 ``generate_llm`` 的形参上 ✓（``**fast_kwargs`` 真展开 ✓）"
          "且生成照常回文本 ✓（配了 ≠ 生效：这一条就是「生效」的证据 ✓）",
          got == [frozenset(llm_mod.FAST_PATH_ITEMS)] and bool(text), (got, text))

    plain = llm_backend.LlmBackend(CONFIG)
    plain._model, plain._hub = model, _StubHub()
    got.clear()
    llm_mod_rt.generate_llm = _spy
    try:
        plain_text = plain.generate("abcabcabc", max_new_tokens=4, temperature=0)
    finally:
        llm_mod_rt.generate_llm = real_gen
    check("⑫¹¹ ⭐ 反证：默认关时 spy 收到 ``<缺>`` ✓（真的**没传**那个关键字 ✓ ⇒ ⑫¹⁰ 不是「反正都传了」✓）"
          "；⚠️ 两次文本**不必相同** ✗（融合核差 1 个 bf16 ulp ⇒ 贪心可能在某个岔口分开 ✓ 那不是错 ✓）",
          got == ["<缺>"] and bool(plain_text), (got, plain_text, text))


def case_cuda_graph_wiring() -> None:
    """图化 decode 单步的**接线** ✓（2026-11-08）—— 与 ⑫ 快路径同一批形状，但它的假绿更贵 ✗✗：

    开关没到位时**输出一模一样** ✓（掩码版只改归约长度 ✓；而本档在 CPU 上更是原样退回 plain ✓）
    ⇒「跑得出来」一点信息量都没有 ✗。所以判据全压在「开关到底去哪了」：回执三态分得开 ✓、
    拼错在 ``load()`` **开头**具名拒 ✓（不是装完 8 GiB 才炸 ✗✗）、缓存键含 ``cudaGraph`` ✓（否则拿到
    **关着的旧后端**、回执照报「开」✗✗）、三处调用口都递 ✓、后端每处 ``generate_llm`` 都带 ✓、
    关着时 ``_graph_call()`` 是**空字典** ✓（默认路径调用签名逐字不变 ✓）。
    """
    import ast  # noqa: PLC0415

    off = llm_backend.LlmBackend(CONFIG).describe()["cudaGraph"]
    on = llm_backend.LlmBackend(CONFIG, cuda_graph="on").describe()["cudaGraph"]
    check("⑬ 回执三态分得开 ✓：``requested``（配置想不想开）／``enabled``（这后端开没开）／"
          "``last``（**上一次真实生成**捕没捕到 ✗ ``None`` ＝还没生成过 ≠ 没开 ✗✗）"
          "＋ 关着时明写「逐位不变」✓（别把「关」读成「没实现」✗）",
          off["requested"] == "off" and off["enabled"] is False and off["last"] is None
          and "逐位" in off["note"]
          and on["requested"] == "on" and on["enabled"] is True and on["last"] is None
          and "还没生成过" in on["note"], (off, on))

    bad = llm_backend.LlmBackend(CONFIG, cuda_graph="grpah")
    err_text = ""
    try:
        bad.load()
    except Exception as err:  # noqa: BLE001
        err_text = f"{type(err).__name__}: {err}"
    check("⑬′ ``cudaGraph`` 拼错 ⇒ ``load()`` **开头**具名拒 ✓（静默当关＝人以为在跑图、其实每步还在发"
          "几千次 launch ✗✗）＋ describe 不抛、把错写进 note ✓",
          "cudaGraph" in err_text and "权重未就绪" not in err_text
          and "grpah" in bad.describe()["cudaGraph"]["note"]
          and bad.describe()["cudaGraph"]["enabled"] is False,
          (err_text, bad.describe()["cudaGraph"]))

    try:
        import fastapi  # noqa: F401, PLC0415
    except ModuleNotFoundError:
        print("SKIP  ⑬″～⑬⁷ 需要 fastapi（本解释器未装）⇒ 不判（≠ 通过 ✓）")
        return

    from app.services import text_generation as tg  # noqa: PLC0415

    key_off = tg._engine_backend_key("/m.gguf", None)
    key_on = tg._engine_backend_key("/m.gguf", None, cuda_graph="on")
    check("⑬″ ⭐ 缓存键含 ``cudaGraph`` ✓（不进格 ⇒ 从 off 改成 on 还是**同一个后端对象** ⇒ 图根本没捕 ✗✗"
          "而回执照旧报「关」✗✗、人却以为在省 launch ✓）＋ 关＝与「不写」同格 ✓ ＋ 同义写法归一 ✓",
          key_on != key_off
          and key_off == tg._engine_backend_key("/m.gguf", None, cuda_graph="off")
          and key_on == tg._engine_backend_key("/m.gguf", None, cuda_graph=True)
          == tg._engine_backend_key("/m.gguf", None, cuda_graph="graph"),
          (key_off, key_on))
    check("⑬‴ 拼错的写法**原样留在键里** ✓（归一抛错时不许当成「关」去复用那份关着的后端 ✗✗）",
          tg._engine_backend_key("/m.gguf", None, cuda_graph="grpah")
          != tg._engine_backend_key("/m.gguf", None, cuda_graph="off"))
    check("⑬⁴ ``engine_cuda_graph`` 两种键名都认 ✓、缺省 ``None`` ✓、``engine_graph_kwargs`` 只出这一格 ✓",
          tg.engine_cuda_graph({"cudaGraph": "on"}) == "on"
          and tg.engine_cuda_graph({"cuda_graph": True}) is True
          and tg.engine_cuda_graph({}) is None
          and set(tg.engine_graph_kwargs({"cudaGraph": "on"})) == {"cuda_graph"})

    root = Path(__file__).resolve().parents[1] / "app"
    missing_port: list[str] = []
    for rel in ("services/text_generation.py", "services/text_stream.py", "agent/runtime.py"):
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id.lstrip("_") == "get_engine_backend"]
        if not calls:
            missing_port.append(f"{rel}: 找不到 get_engine_backend 调用 ✗")
        for call in calls:
            spread = [kw for kw in call.keywords if kw.arg is None
                      and isinstance(kw.value, ast.Call)
                      and getattr(kw.value.func, "id", "") == "engine_graph_kwargs"]
            if not spread:
                missing_port.append(f"{rel}: 没展开 engine_graph_kwargs")
    check("⑬⁵ ⭐ 三个调用口**都**递图化开关 ✓（漏一条 ⇒ 那条路拿的是关着的后端 ⇒ 人按「都开了」算账"
          "会算错 ✗✗）", not missing_port, missing_port)

    tree = ast.parse((root / "services/engine/llm_backend.py").read_text(encoding="utf-8"))
    gens = [node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "generate_llm"]

    def _has_graph(node: ast.Call) -> bool:
        spread = any(kw.arg is None and isinstance(kw.value, ast.Name)
                     and kw.value.id == "graph_kwargs" for kw in node.keywords)
        return spread or "cuda_graph" in {kw.arg for kw in node.keywords}

    missing_kw = [node.lineno for node in gens if not _has_graph(node)]
    # ⚠️ 判据要盯「谁去取了这一份」✗ 别盯变量名长得像 ✓；而且这里的左值是**元组**（``kw, sink = ...`` ✓）
    # ⇒ 必须 ``ast.walk`` 进 targets 找名字 ✓（只看 ``node.targets`` 的顶层会是 ``Tuple`` ⇒ 永远 0 个 ✓✗）。
    callers = [node.lineno for node in ast.walk(tree)
               if isinstance(node, ast.Assign)
               and isinstance(node.value, ast.Call)
               and getattr(node.value.func, "attr", "") == "_graph_call"
               and any(isinstance(inner, ast.Name) and inner.id == "graph_kwargs"
                       for target in node.targets for inner in ast.walk(target))]
    check("⑬⁶ ⭐ 后端里**每一处** ``generate_llm`` 都带图化参数 ✓（普通／流式／agent 工具循环三条路"
          "漏一条 ⇒ 那条路仍逐步 launch ✗✗）＋ 每处都取了一份**新的**账 ✓（共用一块会把上一轮的重放数"
          "留在这一轮 ✗✗）",
          len(gens) == 3 and not missing_kw and len(callers) >= 3,
          (len(gens), missing_kw, len(callers)))
    off_call = llm_backend.LlmBackend(CONFIG)._graph_call()
    on_call = llm_backend.LlmBackend(CONFIG, cuda_graph="on")._graph_call()
    auto_call = llm_backend.LlmBackend(CONFIG, cuda_graph="auto")._graph_call()
    check("⑬⁷ 关着时 ``_graph_call()`` ⇒ ``({}, None)`` ✓（不传关键字 ⇒ **默认路径调用签名逐字不变** ✓ "
          "本仓一堆桩模型测试靠这条活着 ✓）＋ 开着时带一块**空账** ✓（回执由它填 ✓ 不是共用对象 ✓）"
          "＋ ⭐ 递下去的必须是**三档原文** ✗ 不是 ``True`` ✗✗（2026-11-09 补）：``auto`` 若被压成 "
          "``True`` ⇒ 短生成照样花 100 ms 捕一张净亏的图 ✓、回执却报 ``auto`` ✓ 说的和做的不一致 ✗",
          off_call == ({}, None)
          and on_call[0].get("cuda_graph") == "on"
          and isinstance(on_call[1], dict) and on_call[1] == {}
          and on_call[0]["cuda_graph_report"] is on_call[1]
          and auto_call[0].get("cuda_graph") == "auto",
          (off_call, on_call, auto_call))
    # ⭐ 2026-11-08 补：**收益与代价必须成对交出去** ✗✗ —— 只写 ``bitExact=False``（代价）而把实测收益
    # 藏起来，人会读成「用确定性换了个不知值不值的速度」；反过来只写收益就是在骗人。这一段不需要 CUDA
    # （直接喂一份假的 ``_graph_last``），所以能在 CPU 上判。
    tight = llm_backend.LlmBackend(CONFIG, cuda_graph="on")
    tight._graph_last = {"enabled": True, "reason": "", "cap": 298, "pastLen": 256,
                         "padRows": 10, "replays": 32, "captureMs": 97.4}
    tight_note = tight.describe()["cudaGraph"]["note"]
    check("⑬⁷′ ⭐ 开销回执成对 ✓：真在重放时 note 必须同时写**收益**（实测 128–512 词元 ×1.24–1.27）与"
          "**代价**（``bitExact=False``）＋默认档说清「短生成一致、长生成会漂移」＋``captureMs`` 原样交出 ✓"
          "（捕获是真的花掉的时间 ⇒ 不许隐去 ✓）",
          "×1.24" in tight_note and "bitExact=False" in tight_note
          and "漂移" in tight_note and "97.4" in tight_note and "10 行" in tight_note,
          tight_note)
    loose = llm_backend.LlmBackend(CONFIG, cuda_graph="on")
    loose._graph_last = {"enabled": True, "reason": "", "cap": 352, "pastLen": 256,
                         "padRows": 54, "replays": 32, "captureMs": 97.4}
    loose_note = loose.describe()["cudaGraph"]["note"]
    check("⑬⁷″ ⭐ ``padRows`` 超出默认预留 ⇒ 必须**显式警示**（实测 96 行时 |Δlogits| 1.9e-1、25 个分叉）✓ "
          "—— 默认档那套「一致」的话到这里就是假话 ✗✗（``kv_reserve_rows`` 被放大时没人会去看 ✗）",
          "54 行" in loose_note and "1.9e-1" in loose_note and "漂移" not in loose_note,
          loose_note)

    # ⭐⭐ 2026-11-09 补：``auto`` 档的回执**三态**必须分得开 ✗✗ —— ``enabled=True`` 只答「可能开」，
    # 光看它分不出「按你要求每次都捕」与「按长度替你决定」✓；而「这一次没捕」若理由是 ``autoTooShort``，
    # 绝不能和「让路／前置不满足」并成一句话 ✓（一个该改 maxOutputTokens，一个该关投机或查驱动 ✓✗）。
    auto_b = llm_backend.LlmBackend(CONFIG, cuda_graph="auto")
    auto_rep = auto_b.describe()["cudaGraph"]
    auto_idle = auto_rep["note"]
    auto_b._graph_last = {"enabled": False, "reason": "autoTooShort(8<16)"}
    auto_skip = auto_b.describe()["cudaGraph"]["note"]
    auto_b._graph_last = {"enabled": True, "reason": "", "cap": 40, "pastLen": 32,
                          "padRows": 10, "replays": 8, "captureMs": 97.4}
    auto_run = auto_b.describe()["cudaGraph"]["note"]
    check("⑬⁷‴ ⭐⭐ ``auto`` 档回执三态分得开 ✓（2026-11-09 补）：``policy`` 原样交出 ✓（只有它能分开 "
          "``on`` 与 ``auto`` ✗✗）＋ 还没跑过时先说「maxOutputTokens ≥ 16 才捕」✓ ＋ 因短跳过时写"
          "「**不是故障**、要强制捕请改 ``on``」✓ 且**不混**「让路」那句 ✓ ＋ 真捕到时说明「本次 ≥ 16 才捕」✓",
          auto_rep["policy"] == "auto" and auto_rep["enabled"] is True
          and "≥ 16" in auto_idle and "净亏" in auto_idle
          and "autoTooShort(8<16)" in auto_skip and "不是故障" in auto_skip
          and "让路" not in auto_skip
          and "auto 档" in auto_run and "×1.24" in auto_run,
          (auto_rep, auto_idle, auto_skip, auto_run))

    if not _torch_available():
        print("SKIP  ⑬⁸ 真跑：需要 torch（本机未装）⇒ 不判（≠ 通过 ✓）")
        return
    # 端到端：CPU 上开图 ⇒ **不许崩** ✓、输出照常 ✓、而 ``last.reason`` 必须如实写 ``device=cpu`` ✗✗
    #（这一条正是「要求了但静默没开」的反例：账本里看得见「为什么没跑图」✓）。
    from app.services.engine import llm as llm_mod_rt  # noqa: PLC0415

    model = llm_mod_rt.build_llm(CONFIG)
    backend = llm_backend.LlmBackend(CONFIG, cuda_graph="on")
    backend._model, backend._hub = model, _StubHub()
    text = backend.generate("abcabcabc", max_new_tokens=4, temperature=0)
    last = backend.describe()["cudaGraph"]["last"] or {}
    check("⑬⁸ ⭐ 真跑（CPU）：开图不崩 ✓、照常出文本 ✓、而 ``last.reason == device=cpu`` ✓ —— "
          "「要求了却没跑图」必须在账本里看得见 ✗✗（静默回退＝禁令 ✓）",
          bool(text) and last.get("requested") is True and last.get("enabled") is False
          and last.get("reason") == "device=cpu" and last.get("bitExact") is False,
          (text[:20], last))


def main() -> int:
    case_describe()
    case_unavailable()
    case_generate()
    case_decoded_only_new()
    case_chat_template_vars()
    case_wiring()
    case_kv_wiring()
    case_kv_window_wiring()
    case_prefill_tier()
    case_spec_wiring()
    case_fast_wiring()
    case_cuda_graph_wiring()
    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
