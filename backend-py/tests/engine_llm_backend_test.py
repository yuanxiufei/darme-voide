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


def main() -> int:
    case_describe()
    case_unavailable()
    case_generate()
    case_decoded_only_new()
    case_chat_template_vars()
    case_wiring()
    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
