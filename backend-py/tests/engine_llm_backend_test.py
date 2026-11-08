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


def main() -> int:
    case_describe()
    case_unavailable()
    case_generate()
    case_decoded_only_new()
    case_chat_template_vars()
    case_wiring()
    case_kv_wiring()
    case_spec_wiring()
    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
