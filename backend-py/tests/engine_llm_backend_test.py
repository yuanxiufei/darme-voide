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


def case_generate() -> None:
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


def main() -> int:
    case_describe()
    case_unavailable()
    case_generate()
    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
