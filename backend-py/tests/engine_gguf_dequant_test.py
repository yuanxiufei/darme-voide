"""自检：GGUF 权重**反量化**（``engine/gguf_dequant.py`` ✓ 2026-09-25 起 ✓）。

验的是什么 ✗：本仓 :mod:`gguf` 只读元数据 ✗ —— 本套验的是**量化字节 ⇒ fp32 张量**这一步 ✓
（公式照 llama.cpp ``ggml-quants.c`` ✓ MIT ✓）。

判据（都是「看着像装了权重、其实数值全错」的形状 ✓✗）：
1. ⭐⭐ **Q4_K 字节布局 + 6-bit 解包** ✗✗：手造 block（d=1/dmin=0.5/scales 全 0xFF/qs 全 0x11）
   反量化 ⇒ **每个元素 = 31.5** ✓（sc=63、m=63、q=1 ⇒ ``1*63*1 - 0.5*63 = 31.5`` ✓）
   —— 布局或 6-bit 解包错任何一处，这个精确值都对不上 ✗✗；
2. ⭐ **Q8_0** ✓：``d * qs`` ✓；
3. **F16/F32/BF16 直接转换** ✓；
4. ⭐ **未实现的 k-quant（Q6_K）⇒ 具名拒绝** ✗（不静默当 F32 ✗ —— 那会装出「数值全错的权重」✗）。

⚠️ 本套**不宣称**能加载真 Qwen3 GGUF ✗（F32/F16/Q8_0/Q4_K 之外的类型会拒绝 ✓）；
⚠️ 端到端「GGUF 文件 → llm.LlmModel」的权重名映射**未做** ✗（那是下一步 ✓）。

运行::

    ./.venv/Scripts/python.exe tests/engine_gguf_dequant_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="ggdq_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app.services.engine import gguf_dequant as dq  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _raises(fn) -> bool:
    try:
        fn()
    except dq.GgufDequantError:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


def case_direct() -> None:
    data = np.array([1.0, -2.5, 3.25], dtype="<f4").tobytes()
    check("① F32 直接转换 ✓", np.allclose(dq.dequantize_tensor(data, 0, 3), [1.0, -2.5, 3.25]))

    data = np.array([1.0, -2.5, 0.5], dtype="<f2").tobytes()
    check("①′ F16 → fp32 ✓", np.allclose(dq.dequantize_tensor(data, 1, 3), [1.0, -2.5, 0.5]))

    # bf16：1.0=0x3F80、-2.0=0xC000、0.5=0x3F00
    data = np.array([0x3F80, 0xC000, 0x3F00], dtype="<u2").tobytes()
    check("①″ BF16 → fp32 ✓（<<16 左移 ✓）",
          np.allclose(dq.dequantize_tensor(data, 32, 3), [1.0, -2.0, 0.5]))


def case_q8_0() -> None:
    block = np.array([1.0], dtype="<f2").tobytes() + np.arange(32, dtype="<i1").tobytes()
    out = dq.dequantize_tensor(block, 8, 32)
    check("② Q8_0 反量化 = d * qs ✓（d=1 ⇒ 原样 ✓）",
          np.allclose(out, np.arange(32, dtype=np.float32)))

    block = np.array([0.5], dtype="<f2").tobytes() + np.full(32, 4, dtype="<i1").tobytes()
    check("②′ Q8_0 d=0.5 ⇒ 全 2.0 ✓（int8 带符号，可负 ✓）",
          np.allclose(dq.dequantize_tensor(block, 8, 32), 2.0))


def case_q4_k() -> None:
    # ⭐⭐ 手造 block：d=1.0、dmin=0.5、scales 全 0xFF（⇒ 每个 6-bit scale=min=63 ✓）、
    #    qs 全 0x11（⇒ 每个 4-bit q=1 ✓）⇒ y = 1*63*1 - 0.5*63 = 31.5 ✓
    block = (np.array([1.0, 0.5], dtype="<f2").tobytes()
             + bytes([0xFF] * 12) + bytes([0x11] * 128))
    out = dq.dequantize_tensor(block, 12, 256)
    check("③ ⭐⭐ Q4_K 字节布局 + 6-bit 解包 ✓（全 256 元素 == 31.5 ✓✗）",
          out.shape == (256,) and np.allclose(out, 31.5),
          (float(out[0]), float(out[128]), float(out[-1])))

    # 换个 qs：低 4 bit=2、高 4 bit=3 ⇒ 前半 y=1*63*2-31.5=94.5、后半 y=1*63*3-31.5=157.5
    block = (np.array([1.0, 0.5], dtype="<f2").tobytes()
             + bytes([0xFF] * 12) + bytes([0x32] * 128))
    out = dq.dequantize_tensor(block, 12, 256)
    check("③′ Q4_K 低/高 4-bit 分派正确 ✓（前半 94.5 / 后半 157.5 ✓）",
          np.allclose(out[:32], 94.5) and np.allclose(out[32:64], 157.5)
          and np.allclose(out[192:224], 94.5) and np.allclose(out[224:256], 157.5),
          (float(out[0]), float(out[32])))


def case_unsupported() -> None:
    check("④ ⭐ 未实现的 Q6_K ⇒ 具名拒绝 ✗（不静默当 F32 ✗）",
          _raises(lambda: dq.dequantize_tensor(b"\x00" * 210, 14, 256)))
    check("④′ 块不整除（Q4_K numel=128）⇒ 拒 ✗",
          _raises(lambda: dq.dequantize_tensor(b"\x00" * 144, 12, 128)))


def main() -> int:
    case_direct()
    case_q8_0()
    case_q4_k()
    case_unsupported()
    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
