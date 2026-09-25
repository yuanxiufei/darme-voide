"""GGUF 权重**反量化**（2026-09-25 起）。

## 为什么单独一层

:mod:`gguf` 只读**元数据**（头部 + 张量表 + 字节数）✗，**明确不做反量化** ✗（见它的模块头 ✓）。
本模块补上「把 GGUF 数据区里的量化字节 ⇒ fp32 张量」这一步 ✓ —— Qwen3 / Llama 的 GGUF 权重
从此能**真装上**本仓的 :mod:`llm` ✓。

## 事实来源（不猜 ✗）

llama.cpp 的 ``ggml-quants.c`` ✓（**MIT** 许可 ✓ —— 公式可照实现 ✓）。逐条核对过 ✓：
``Q8_0`` 每 32 元素 = f16 ``d`` + 32×int8 ✓；``Q4_K`` 每 256 元素 = f16 ``d``/``dmin`` +
12 字节 6-bit ``scales``（8 个 scale + 8 个 min ✓ 交织存 ✓）+ 128 字节 4-bit ``qs`` ✓。

## 支持范围

* F32 / F16 / BF16：直接转换 ✓；
* **Q8_0** ✓ / **Q4_K** ✓（Qwen3 最常用 ✓）；
* ⚠️ 其余 k-quant（Q2_K / Q3_K / Q5_K / Q6_K ✓）**后续补** ✗ —— 块布局已在
  :data:`gguf.TENSOR_TYPES` 核清 ✓，但反量化公式未实现 ⇒ 遇到就**具名拒绝** ✗（不静默当 F32 ✗）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import gguf as gguf_mod

__all__ = ["SUPPORTED_DEQUANT", "dequantize_tensor", "load_weights"]


class GgufDequantError(RuntimeError):
    """张量反量化不了 ✓（未知/未实现的量化类型 ✓）。"""


#: 已实现反量化的张量类型 id → 名字 ✓（⚠️ 这是「能反量化」的白名单 ✗，不是全部 ✓）
SUPPORTED_DEQUANT: dict[int, str] = {
    0: "F32", 1: "F16", 32: "BF16", 8: "Q8_0", 12: "Q4_K",
}


def _dequant_f32(data: bytes, numel: int) -> np.ndarray:
    return np.frombuffer(data, dtype="<f4", count=numel).astype(np.float32)


def _dequant_f16(data: bytes, numel: int) -> np.ndarray:
    return np.frombuffer(data, dtype="<f2", count=numel).astype(np.float32)


def _dequant_bf16(data: bytes, numel: int) -> np.ndarray:
    raw = np.frombuffer(data, dtype="<u2", count=numel)
    return (raw.astype(np.uint32) << 16).view(np.float32)


def _dequant_q8_0(data: bytes, numel: int) -> np.ndarray:
    """Q8_0 ✓：每 32 元素 = f16 ``d`` + 32×int8 ✓ ⇒ ``y = d * qs`` ✓。"""
    if numel % 32 != 0:
        raise GgufDequantError(f"Q8_0 块不整除：numel={numel} ✗")
    out = np.empty(numel, dtype=np.float32)
    for index in range(numel // 32):
        scale = float(np.frombuffer(data, dtype="<f2", count=1, offset=index * 34)[0])
        qs = np.frombuffer(data, dtype="<i1", count=32, offset=index * 34 + 2)
        out[index * 32:(index + 1) * 32] = qs.astype(np.float32) * scale
    return out


def _get_scale_min_k4(j: int, scales: np.ndarray) -> tuple[int, int]:
    """Q4_K 的 6-bit 解包 ✓（8 个 scale + 8 个 min，交织存进 12 字节 ✓ —— 公式照 llama.cpp ✓）。"""
    if j < 4:
        return int(scales[j] & 63), int(scales[j + 4] & 63)
    return (int(scales[j + 4] & 0xF) | (int(scales[j - 4] >> 6) << 4),
            int(scales[j + 4] >> 4) | (int(scales[j - 0] >> 6) << 4))


def _dequant_q4_k(data: bytes, numel: int) -> np.ndarray:
    """Q4_K ✓：每 256 元素 = 2×f16 + 12 字节 scales + 128 字节 4-bit ✓。

    反量化 ✓：每 64 元素用两个 6-bit scale/min ⇒ ``y = d*sc * q - dmin*m`` ✓。
    """
    if numel % 256 != 0:
        raise GgufDequantError(f"Q4_K 块不整除：numel={numel} ✗")
    out = np.empty(numel, dtype=np.float32)
    for index in range(numel // 256):
        base = index * 144
        d = float(np.frombuffer(data, dtype="<f2", count=1, offset=base)[0])
        dmin = float(np.frombuffer(data, dtype="<f2", count=1, offset=base + 2)[0])
        scales = np.frombuffer(data, dtype="<u1", count=12, offset=base + 4)
        qs = np.frombuffer(data, dtype="<u1", count=128, offset=base + 16)
        is_ = 0
        for j in range(0, 256, 64):
            sc0, m0 = _get_scale_min_k4(is_, scales)
            sc1, m1 = _get_scale_min_k4(is_ + 1, scales)
            d1, m1v = d * sc0, dmin * m0
            d2, m2v = d * sc1, dmin * m1
            q = qs[(j // 64) * 32:(j // 64) * 32 + 32]
            out[index * 256 + j:index * 256 + j + 32] = \
                (q & 0xF).astype(np.float32) * d1 - m1v
            out[index * 256 + j + 32:index * 256 + j + 64] = \
                (q >> 4).astype(np.float32) * d2 - m2v
            is_ += 2
    return out


_DEQUANT_FN = {0: _dequant_f32, 1: _dequant_f16, 32: _dequant_bf16,
               8: _dequant_q8_0, 12: _dequant_q4_k}


def dequantize_tensor(data: bytes, type_id: int, numel: int) -> np.ndarray:
    """一块量化字节 ⇒ fp32 数组 ✓（未知/未实现的类型 ⇒ **具名拒绝** ✗）。"""
    fn = _DEQUANT_FN.get(int(type_id))
    if fn is None:
        name = gguf_mod.TENSOR_TYPES.get(int(type_id), (f"id={type_id}",))[0]
        raise GgufDequantError(
            f"GGUF 张量类型 {name} 的反量化**未实现** ✗ ⇒ 本模块**不静默当 F32** ✗"
            f"（已支持：{' / '.join(sorted(SUPPORTED_DEQUANT.values()))} ✓）")
    return fn(data, int(numel))


def load_weights(path: str | Path, info: Any = None) -> dict[str, np.ndarray]:
    """把整个 GGUF 文件的权重反量化成 ``{张量名: fp32 数组}`` ✓。

    ⚠️ 大文件（Qwen3 14B ≈ 9 GiB）会**整份读进内存 + 反量化成 fp32** ✗（≈ 17 GiB ✓）——
    真机按需加载时用 :func:`dequantize_tensor` **逐张量**读 ✓，别一次全提 ✗。
    """
    info = info or gguf_mod.inspect(path)
    with open(path, "rb") as handle:
        weights: dict[str, np.ndarray] = {}
        for name, tensor in info.tensors.items():
            handle.seek(info.data_start + tensor.offset)
            blob = handle.read(tensor.nbytes)
            if len(blob) != tensor.nbytes:
                raise GgufDequantError(f"张量 {name} 数据读不全（{len(blob)}/{tensor.nbytes}）✗")
            weights[name] = dequantize_tensor(blob, tensor.type_id, tensor.numel)
        return weights
