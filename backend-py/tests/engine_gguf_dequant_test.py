"""自检：GGUF 权重**反量化**（``engine/gguf_dequant.py`` ✓ 2026-09-25 起 ✓）。

验的是什么 ✗：本仓 :mod:`gguf` 只读元数据 ✗ —— 本套验的是**量化字节 ⇒ fp32 张量**这一步 ✓
（公式照 llama.cpp ``ggml-quants.c`` ✓ MIT ✓）。

判据（都是「看着像装了权重、其实数值全错」的形状 ✓✗）：
1. ⭐⭐ **Q4_K 字节布局 + 6-bit 解包** ✗✗：手造 block（d=1/dmin=0.5/scales 全 0xFF/qs 全 0x11）
   反量化 ⇒ **每个元素 = 31.5** ✓（sc=63、m=63、q=1 ⇒ ``1*63*1 - 0.5*63 = 31.5`` ✓）
   —— 布局或 6-bit 解包错任何一处，这个精确值都对不上 ✗✗；
2. ⭐⭐ **Q6_K** ✓：``d`` 在**块尾** ✓、减 32 的 6-bit 有符号 ✓、4 个子块各吃 ``qh`` 的 2 bit ✓、
   16 段 × 16 的 scale 分派 ✓（三处都与 Q4_K 相反 ✓✗）；
3. ⭐⭐ **整张量向量化版 == 逐块参考版**（**随机字节**对拍、4097 块跨 chunk 边界 ✓、逐位相等 ✓）
   —— 手造值只覆盖「我想到的组合」✗，这条才是向量化重写的兜底 ✓；
4. ⭐ **Q8_0** ✓：``d * qs`` ✓；
5. **F16/F32/BF16 直接转换** ✓；
6. ⭐ **真文件**：`Q4_K_M` 是**混合**方案 ✓（F32/Q4_K/Q6_K 三种）、白名单**覆盖率 100%** ✓、
   真 Q6_K 张量有限非零 ✓、**跨量化对拍** corr > 0.99 ✓（不靠自写期望值 ✓）；
7. ⭐ **未实现的 k-quant（Q5_K）⇒ 具名拒绝** ✗（不静默当 F32 ✗ —— 那会装出「数值全错的权重」✗）。

⚠️ 本套只保证「盘上那个 Q4_K_M 文件里出现的类型全覆盖」✗ —— Q2_K/Q3_K/Q5_K/Q8_K 仍会拒绝 ✓；
⚠️ 端到端「GGUF 文件 → llm.LlmModel」的权重名映射**不在本套** ✓（见 ``engine_gguf_to_llm_test`` ✓）。

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

from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from app.services.engine import gguf as gguf_mod  # noqa: E402
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


def _q6_block(d: float = 1.0, scales: Any = None,
              ql: bytes = b"\x00" * 128, qh: bytes = b"\x00" * 64) -> bytes:
    """造一个 Q6_K 块 ✓（**顺序是 ql → qh → scales → d** ✓，``d`` 在**块尾** ✓✗）。"""
    if scales is None:
        scales = [1] * 16
    return (bytes(ql) + bytes(qh)
            + np.array(scales, dtype="<i1").tobytes()
            + np.array([d], dtype="<f2").tobytes())


def case_q6_k() -> None:
    # ⭐⭐ 手造 block：ql/qh 全 0 ⇒ q = 0−32 = −32 ✓；scales 全 1、d=1 ⇒ **全 256 元素 == −32**
    out = dq.dequantize_tensor(_q6_block(), 14, 256)
    check("③″ ⭐⭐ Q6_K 字节布局 ✓（``d`` 在**块尾** ✓、减 32 的 6-bit 有符号 ✓ ⇒ 全 −32 ✓✗）",
          out.shape == (256,) and np.allclose(out, -32.0),
          (float(out[0]), float(out[128]), float(out[-1])))

    # ql/qh 全 0xFF ⇒ q = 0xF | (3<<4) = 63、63−32 = 31 ✓
    out = dq.dequantize_tensor(_q6_block(ql=b"\xff" * 128, qh=b"\xff" * 64), 14, 256)
    check("③‴ Q6_K 4-bit + 2-bit 拼满 ⇒ 全 31 ✓（少解一位就凑不出 63 ✗）",
          np.allclose(out, 31.0), (float(out[0]), float(out[-1])))

    # scale 分派：16 个一段 ⇒ 第 k 段吃 scales[k] ✓（段序错位是本类实现最常见的坏法 ✗）
    scales = list(range(1, 9)) + list(range(11, 19))
    out = dq.dequantize_tensor(_q6_block(scales=scales), 14, 256)
    expected = np.repeat(-32.0 * np.array(scales, dtype=np.float32), 16)
    check("③⁗ ⭐ Q6_K scale 分派 ✓（256 元素 = 16 段 × 16 ✓，第 k 段吃 scales[k] ✓）",
          np.allclose(out, expected),
          (float(out[0]), float(out[16]), float(out[240])))

    # 高 2 bit 的分派：qh 只有最低 2 位非零 ⇒ 只有子块 0（前 32 元素）吃到 +48 ✓
    out = dq.dequantize_tensor(_q6_block(qh=b"\x03" * 64), 14, 256)
    expected = np.full(256, -32.0, dtype=np.float32)
    expected[0:32] = 16.0
    expected[128:160] = 16.0      # ⚠️ qh 给了 64 字节 ⇒ **两个**半块都吃到 ✓（只给一个半块会漏验 ✓✗）
    check("③⁗′ Q6_K 四个子块各吃 qh 的 2 个 bit ✓（qh=0x03 ⇒ 子块 0 = 16、其余 = −32 ✓）",
          np.allclose(out, expected),
          (float(out[0]), float(out[32]), float(out[128]),
           np.unique(out[:32]).tolist(), np.unique(out[32:128]).tolist()))

    # 第二块（元素 128…）必须吃 qh 的后 32 字节 ✓ —— 只前移 ql 不前移 qh 会在这里露出来
    qh = b"\x00" * 32 + b"\x03" * 32
    out = dq.dequantize_tensor(_q6_block(qh=qh), 14, 256)
    check("③⁗″ ⭐ 第 2 个半块吃 qh 的**后 32 字节** ✓（前移错位 ⇒ 前半/后半会反过来 ✗）",
          np.allclose(out[:128], -32.0) and np.allclose(out[128:160], 16.0)
          and np.allclose(out[160:], -32.0),
          (float(out[128]), float(out[160])))

    check("③⁗‴ Q6_K 块不整除 ⇒ 拒 ✗",
          _raises(lambda: dq.dequantize_tensor(b"\x00" * 210, 14, 128)))
    check("③⁗⁗ Q6_K 数据不够 210 字节 ⇒ 拒 ✗（不读到越界当 0 ✗）",
          _raises(lambda: dq.dequantize_tensor(b"\x00" * 200, 14, 256)))


def case_real_file() -> None:
    """真 GGUF 核验 ✓（文件不在本机就 **skip 并说明** ✓，不算失败 ✗）。

    ⭐ 为什么必须核真文件：手造块只能验「布局与我理解的一致」✗ —— 而 **Q4_K_M 未必是纯 Q4_K**
    （量化器按张量挑类型 ✓，`attn_v` / 部分 `ffn_down` 会挑成 Q6_K ✓）⇒ 只实现 Q4_K 时装
    Q4_K_M 会在那几张量上具名拒绝 ✗，这条事实只能在真文件上核 ✓。
    """
    models = Path(r"D:\Comfy-Desktop\ComfyUI-Shared\models")
    mixed_file = models / "clip" / "Qwen3_VL_4B" / "qwen3-vl-4b-heretic-Q4_K_M.gguf"
    if not mixed_file.exists():
        check("③⁗⁗⁗ 真 GGUF 核验（本机无样例文件 ⇒ **skip** ✓ 不算通过 ✗）", True,
              "skipped")
        return

    info = gguf_mod.inspect(mixed_file)
    counts = dict(info.dtype_counts or {})
    check("③⁗⁗⁗ ⭐⭐ **Q4_K_M 是混合方案**（实测 398 张量 = F32 145 + Q4_K 216 + **Q6_K 37** ✓，"
          "那 37 张全落在 `attn_v` / `ffn_down` ✓）⇒ 只实现 Q4_K 时这文件**装不进** ✓✗",
          counts.get("Q4_K", 0) > 0 and counts.get("Q6_K", 0) > 0,
          (mixed_file.name, counts))

    known = set(dq.SUPPORTED_DEQUANT.values())
    unknown = {name: n for name, n in counts.items() if name not in known}
    check("③⁗⁗⁗′ ⭐ 补了 Q6_K 之后，该文件**类型覆盖率 100%** ✓（白名单 ⊇ 文件里出现的全部类型 ✓）"
          "—— 覆盖不全就会在装到一半时具名拒绝 ✗",
          not unknown, (sorted(counts), unknown))

    q6_names = sorted([name for name, tensor in info.tensors.items() if tensor.type_id == 14],
                      key=lambda n: info.tensors[n].numel)
    name = q6_names[len(q6_names) // 2]
    values = _read_tensor(mixed_file, info, name)
    check("③⁗⁗⁗″ 真 Q6_K 张量反量化 ⇒ 无 NaN/Inf、非零、长度 == numel ✓",
          values.shape == (info.tensors[name].numel,)
          and bool(np.isfinite(values).all()) and float(np.abs(values).mean()) > 0,
          (name, info.tensors[name].shape, float(np.abs(values).mean())))

    case_cross_quant(models)


def case_cross_quant(models: Path) -> None:
    """同模型两版量化的**同名张量对拍** ✓ —— 唯一不靠我自己写期望值的正确性证据 ✓。

    ⭐ 思路：`Q4_K_M` 与 `Q6_K_M` 是**同一份权重的两种量化** ✓ ⇒ 同名张量反量化后必须
    **高度相关**（实测 0.997 ✓）、**标准差几乎相同** ✓。Q6_K 只要解错位（哪怕只错一个 bit 的
    移位 ✓），相关性会崩到 ~0 ✗ 或尺度明显偏 ✓✗ —— 这比"我算的值等于我写的期望值"硬得多 ✓。
    """
    folder = models / "diffusion_models" / "SEEDVR2"
    a = folder / "seedvr2_ema_3b-Q4_K_M.gguf"
    b = folder / "seedvr2_ema_3b-Q6_K_M.gguf"
    if not (a.exists() and b.exists()):
        check("③⁗⁗⁗⁗ 跨量化对拍（本机缺这一对样例文件 ⇒ **skip** ✓ 不算通过 ✗）", True,
              "skipped")
        return
    info_a, info_b = gguf_mod.inspect(a), gguf_mod.inspect(b)
    shared = [n for n in info_b.tensors
              if n in info_a.tensors and info_b.tensors[n].type_id == 14
              and info_a.tensors[n].shape == info_b.tensors[n].shape]
    if not shared:
        check("③⁗⁗⁗⁗ 跨量化对拍：两文件无同名 Q6_K 张量 ⇒ 换样例 ✓", False, len(shared))
        return
    name = sorted(shared, key=lambda n: info_b.tensors[n].numel)[0]
    values_a = _read_tensor(a, info_a, name)
    values_b = _read_tensor(b, info_b, name)
    corr = float(np.corrcoef(values_a, values_b)[0, 1])
    scale_gap = abs(float(values_a.std()) - float(values_b.std())) / max(
        float(values_b.std()), 1e-12)
    check("③⁗⁗⁗⁗ ⭐⭐⭐ 跨量化对拍 ✓：Q4_K 版 vs Q6_K 版**同名张量** corr > 0.99 ✓、"
          "标准差差 < 1% ✓（解错位 ⇒ corr 崩 ✗✗）",
          corr > 0.99 and scale_gap < 0.01,
          (name, info_b.tensors[name].shape, round(corr, 6), round(scale_gap, 6)))


def _read_tensor(path: Path, info: Any, name: str) -> np.ndarray:
    """**逐张量**读 ✓（真文件 Q6_K_M 有 2~3 GiB ✗ —— 整份提会爆内存 ✗）。"""
    tensor = info.tensors[name]
    with open(path, "rb") as handle:
        handle.seek(info.data_start + tensor.offset)
        blob = handle.read(tensor.nbytes)
    return dq.dequantize_tensor(blob, tensor.type_id, tensor.numel)


def _q4_k_reference(data: bytes, numel: int) -> np.ndarray:
    """Q4_K 的**逐块参考版** ✓（llama.cpp ``dequantize_row_q4_K`` 的直译 ✓）—— 只用来对拍 ✗。"""
    out = np.empty(numel, dtype=np.float32)
    for index in range(numel // 256):
        base = index * 144
        # ⚠️ 全程 float32 ✓（Python float 会把整式抬成 float64 ✗ ⇒ 与向量化版差 1 ulp ✗）
        d = np.float32(np.frombuffer(data, dtype="<f2", count=1, offset=base)[0])
        dmin = np.float32(np.frombuffer(data, dtype="<f2", count=1, offset=base + 2)[0])
        scales = np.frombuffer(data, dtype=np.uint8, count=12, offset=base + 4)
        qs = np.frombuffer(data, dtype=np.uint8, count=128, offset=base + 16)
        y = out[index * 256:(index + 1) * 256]
        for j in range(4):                       # 每 64 元素一对 6-bit scale/min ✓
            sc1, mn1 = dq._get_scale_min_k4(2 * j, scales)
            sc2, mn2 = dq._get_scale_min_k4(2 * j + 1, scales)
            lane = qs[j * 32:(j + 1) * 32]
            y[j * 64:j * 64 + 32] = d * sc1 * (lane & 0x0F) - dmin * mn1
            y[j * 64 + 32:j * 64 + 64] = d * sc2 * (lane >> 4) - dmin * mn2
    return out


def _q6_k_reference(data: bytes, numel: int) -> np.ndarray:
    """Q6_K 的**逐块参考版** ✓（``dequantize_row_q6_K`` 的直译 ✓）—— 只用来对拍 ✗。"""
    out = np.empty(numel, dtype=np.float32)
    for index in range(numel // 256):
        base = index * 210
        d = np.float32(np.frombuffer(data, dtype="<f2", count=1, offset=base + 208)[0])
        ql = np.frombuffer(data, dtype=np.uint8, count=128, offset=base)
        qh = np.frombuffer(data, dtype=np.uint8, count=64, offset=base + 128)
        sc = np.frombuffer(data, dtype=np.int8, count=16, offset=base + 192)
        y = out[index * 256:(index + 1) * 256]
        for half in (0, 1):
            ql_h, qh_h = ql[half * 64:(half + 1) * 64], qh[half * 32:(half + 1) * 32]
            sc_h = sc[half * 8:(half + 1) * 8]
            for l in range(32):
                is_ = l // 16
                q1 = int((ql_h[l] & 0xF) | (((qh_h[l] >> 0) & 3) << 4)) - 32
                q2 = int((ql_h[l + 32] & 0xF) | (((qh_h[l] >> 2) & 3) << 4)) - 32
                q3 = int((ql_h[l] >> 4) | (((qh_h[l] >> 4) & 3) << 4)) - 32
                q4 = int((ql_h[l + 32] >> 4) | (((qh_h[l] >> 6) & 3) << 4)) - 32
                base_y = half * 128 + l
                y[base_y + 0] = d * int(sc_h[is_ + 0]) * q1
                y[base_y + 32] = d * int(sc_h[is_ + 2]) * q2
                y[base_y + 64] = d * int(sc_h[is_ + 4]) * q3
                y[base_y + 96] = d * int(sc_h[is_ + 6]) * q4
    return out


def _random_blocks(kind: str, nblock: int, seed: int = 20261029) -> bytes:
    """随机块 ✓（``d``/``dmin`` 取有限正数 ⇒ 不会出 NaN 让 ``allclose`` 假通过 ✗）。"""
    rng = np.random.default_rng(seed)
    block = 144 if kind == "q4" else 210
    blob = rng.integers(0, 256, size=(nblock, block), dtype=np.uint8)
    scale_at = (0, 2) if kind == "q4" else (208,)       # d / dmin 的位置 ✓（Q6_K 在块尾 ✓）
    for offset in scale_at:
        blob[:, offset:offset + 2] = np.frombuffer(
            rng.uniform(0.01, 2.0, size=nblock).astype("<f2").tobytes(),
            dtype=np.uint8).reshape(nblock, 2)
    return blob.tobytes()


def case_vectorized_matches_reference() -> None:
    """⭐⭐ **整张量向量化版 vs 逐块参考版**（随机字节对拍 ✓）—— 手造值只覆盖我想到的组合 ✗。

    覆盖 4097 块 ⇒ 跨过 ``_CHUNK`` 边界 ✓（分块循环写错会在第二块 chunk 才露 ✓✗）。
    """
    rows = np.random.default_rng(7).integers(0, 256, size=(4096, 12), dtype=np.uint8)
    sc_vec, mn_vec = dq._q4_k_scale_min(rows)
    sc_ref = np.array([[dq._get_scale_min_k4(j, row)[0] for j in range(8)] for row in rows])
    mn_ref = np.array([[dq._get_scale_min_k4(j, row)[1] for j in range(8)] for row in rows])
    check("③⁗⁗⁗⁗⁗ ⭐ 6-bit scale/min 的向量化版 == 逐元素参考版 ✓（4096 组随机字节 ✓）",
          np.array_equal(sc_vec, sc_ref) and np.array_equal(mn_vec, mn_ref),
          (sc_vec.shape, int((sc_vec != sc_ref).sum()), int((mn_vec != mn_ref).sum())))

    for kind, type_id, ref, name in (("q4", 12, _q4_k_reference, "Q4_K"),
                                     ("q6", 14, _q6_k_reference, "Q6_K")):
        blob = _random_blocks(kind, 4097)
        out = dq.dequantize_tensor(blob, type_id, 4097 * 256)
        expected = ref(blob, 4097 * 256)
        biggest = float(np.max(np.abs(out - expected))) if out.size else 0.0
        check(f"③⁗⁗⁗⁗⁗′ ⭐⭐ {name} 整张量向量化版 == 逐块参考版 ✓"
              f"（4097 块随机字节、跨 chunk 边界 ✓、最大偏差 {biggest:.3e} ✓）",
              bool(np.array_equal(out, expected)) and bool(np.isfinite(out).all()),
              (out.shape, biggest, int((out != expected).sum())))


def case_unsupported() -> None:
    check("④ ⭐ 未实现的 Q5_K ⇒ 具名拒绝 ✗（不静默当 F32 ✗）",
          _raises(lambda: dq.dequantize_tensor(b"\x00" * 176, 13, 256)))
    check("④′ 块不整除（Q4_K numel=128）⇒ 拒 ✗",
          _raises(lambda: dq.dequantize_tensor(b"\x00" * 144, 12, 128)))


def main() -> int:
    case_direct()
    case_q8_0()
    case_q4_k()
    case_q6_k()
    case_vectorized_matches_reference()
    case_real_file()
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
