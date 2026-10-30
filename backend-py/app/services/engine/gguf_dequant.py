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
* **Q8_0** ✓ / **Q4_K** ✓ / **Q6_K** ✓（2026-10-29 补 ✓）；
* ⚠️ 其余 k-quant（Q2_K / Q3_K / Q5_K）**后续补** ✗ —— 块布局已在
  :data:`gguf.TENSOR_TYPES` 核清 ✓，但反量化公式未实现 ⇒ 遇到就**具名拒绝** ✗（不静默当 F32 ✗）。

### 为什么 Q6_K 是「非补不可」的那个（2026-10-29）

`Q4_K_M` **可能**是混合方案 ✓ —— 不是「全部 Q4_K」✓：量化器**按张量挑类型** ✓，
`attn_v` / 部分 `ffn_down` 这类敏感张量会挑成 **Q6_K** ✓（M/S 后缀只在
``general.file_type`` 里，张量表里看到的就是块类型 ✓）。
所以只实现 Q4_K 时，**一个 Q4_K_M 文件照样装不上** ✗（撞在那几张量上 ✓），
而且报错只在「读到那几个张量」时才出现 ✓ —— 症状是「前面几十个张量都对、突然具名拒绝」✓✗。

盘上实测：`qwen3-vl-4b-heretic-Q4_K_M.gguf` = **398 张量**（F32 145 / Q4_K 216 / **Q6_K 37** ✓，
那 37 张全是 `blk.N.attn_v.weight` 与 `blk.N.ffn_down.weight` ✓）；补上 Q6_K 后该文件
**类型覆盖率 100%** ✓（白名单 ⊇ 文件里出现的全部类型 ✓）。

⚠️ 反过来也**不能凭后缀断定** ✗：`seedvr2_ema_3b-Q4_K_M.gguf` 实测是**纯 Q4_K**
（210 张量、无一个 Q6_K ✓）—— 量化器的张量名单没命中该架构的命名，就全是 Q4_K ✓。
所以「M 后缀 ⇒ 一定含 Q6_K」是错的口径 ✓✗；判据只能是**读文件里的实际块类型** ✓。
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
    0: "F32", 1: "F16", 32: "BF16", 8: "Q8_0", 12: "Q4_K", 14: "Q6_K",
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


def _q4_k_scale_min(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n, 12) uint8 ⇒ ``sc`` (n, 8) / ``mn`` (n, 8) ✓ —— **向量化版的** ``_get_scale_min_k4`` ✓。

    两版必须**逐元素等价** ✓（逐元素那版 :func:`_get_scale_min_k4` 保留下来当参考 ✓，
    ``engine_gguf_dequant_test`` 用随机字节对拍两版 ✓ —— 手造值只覆盖我想到的组合 ✗）。
    """
    count = scales.shape[0]
    sc = np.empty((count, 8), dtype=np.uint8)
    mn = np.empty((count, 8), dtype=np.uint8)
    sc[:, 0:4] = scales[:, 0:4] & 63
    mn[:, 0:4] = scales[:, 4:8] & 63
    sc[:, 4:8] = (scales[:, 8:12] & 0x0F) | ((scales[:, 0:4] >> 6) << 4)
    mn[:, 4:8] = (scales[:, 8:12] >> 4) | ((scales[:, 4:8] >> 6) << 4)
    return sc, mn


#: Q4_K 每块字节数 ✓（d[2] + dmin[2] + scales[12] + qs[128] ✓）
_Q4_K_BLOCK = 144
#: 一次向量化处理多少块 ✓ —— 只影响峰值内存 ✓（4096 块 ≈ 576 KiB 输入 / 4 MiB 输出 ✓），不影响结果 ✓
_Q4_K_CHUNK = 4096

#: Q4_K 的 256 列 → 字节下标 / 是否取低半字节 / 段号 ✓（模块级预计算 ✓，每块不必重算 ✓）
_Q4_K_COL = np.arange(256)
_Q4_K_BYTE = ((_Q4_K_COL % 32) + 32 * (_Q4_K_COL // 64)).astype(np.intp)
_Q4_K_IS_LOW = (_Q4_K_COL % 64) < 32
_Q4_K_SEG = (_Q4_K_COL // 32).astype(np.intp)


def _dequant_q4_k(data: bytes, numel: int) -> np.ndarray:
    """Q4_K ✓：每 256 元素 = d(f16) + dmin(f16) + 12 字节 scales + 128 字节 4-bit ✓（144 字节/块 ✓）。

    反量化 ✓：``y = d*sc*lane − dmin*mn`` ✓ —— 256 元素 = 8 段 × 32 ✓，每 64 元素用两个 6-bit
    scale/min（段号 = 列 ÷ 32 ✓；低半字节取偶数段 ✓、高半字节取奇数段 ✓）。

    ⚠️ **整张量向量化**（2026-10-29 改 ✓）：原实现是「per-block Python 循环 + 每段两次切片赋值」✓✗
    —— 真模型 3138 M 元素实测只有 **11.5 M 元素/秒 ⇒ 单份 4B 权重光反量化 273 秒** ✗。
    向量化后真文件 `blk.27.attn_output.weight`（10.5 M 元素）实测 **100.6 M 元素/秒** ✓（8.7× ✓），
    4B 整份估算 **≈31 秒** ✓（Q6_K 同法 ✓ 157.2 M 元素/秒 / 30.8× ✓）。
    等价性由 `engine_gguf_dequant_test` 的「随机字节 × 两版对拍」保证 ✓（不靠手造值 ✗）——
    ⚠️ 对拍能成立的前提是**结合顺序也一样** ✓：下面 `d * sc * q_val` 与 llama.cpp 的
    ``d*sc*q`` 同序 ✓，改成 ``q*sc*d`` 会差 1 ulp ✗，逐位对拍立刻报 ✓。
    """
    if numel % 256 != 0:
        raise GgufDequantError(f"Q4_K 块不整除：numel={numel} ✗")
    nblock = numel // 256
    need = nblock * _Q4_K_BLOCK
    if need > len(data):
        raise GgufDequantError(f"Q4_K 数据不够：需要 {need} 字节、只有 {len(data)} ✗")
    raw = np.frombuffer(data, dtype=np.uint8, count=need).reshape(nblock, _Q4_K_BLOCK)
    out = np.empty((nblock, 256), dtype=np.float32)
    for start in range(0, nblock, _Q4_K_CHUNK):
        stop = min(start + _Q4_K_CHUNK, nblock)
        chunk = raw[start:stop]
        # ⚠️ 切出来的列不是 C 连续 ⇒ view 成 f16 前必须 ascontiguousarray ✓（同 Q6_K ✓）
        d = np.ascontiguousarray(chunk[:, 0:2]).view("<f2").astype(np.float32).reshape(-1, 1)
        dmin = np.ascontiguousarray(chunk[:, 2:4]).view("<f2").astype(np.float32).reshape(-1, 1)
        sc, mn = _q4_k_scale_min(chunk[:, 4:16])
        qs = chunk[:, 16:144]
        lanes = qs[:, _Q4_K_BYTE]                        # (n, 256) uint8 ✓
        q_val = np.where(_Q4_K_IS_LOW, lanes & 0x0F, lanes >> 4).astype(np.float32)
        out[start:stop] = (d * sc[:, _Q4_K_SEG] * q_val
                           - dmin * mn[:, _Q4_K_SEG])
    return out.reshape(-1)


#: Q6_K 每块字节数 ✓（ql[128] + qh[64] + scales[16] + d[2] ✓）
_Q6_K_BLOCK = 210
#: 一次向量化处理多少块 ✓ —— 只影响峰值内存 ✓（4096 块 ≈ 860 KiB 输入 / 4 MiB 输出 ✓），不影响结果 ✓
_Q6_K_CHUNK = 4096


def _dequant_q6_k(data: bytes, numel: int) -> np.ndarray:
    """Q6_K ✓：每 256 元素 = ``ql[128]`` + ``qh[64]`` + ``scales[16]`` + ``d``(f16) ✓。

    ⚠️ **``d`` 在块尾**（offset 208 ✓）—— 与 Q4_K/Q5_K 的 ``d`` 打头**相反** ✗：按 Q4_K 的习惯
    从头读 ``d``，读到的会是 ``ql[0..1]`` ✓✗，产物是「数值全错但形状对」✓✗。

    解包（公式照 llama.cpp ``dequantize_row_q6_K`` ✓）：

    * 低 4 bit 在 ``ql`` ✓、高 2 bit 在 ``qh`` ✓ ⇒ ``q = (ql & 0xF) | (((qh >> 2k) & 3) << 4)`` ✓，
      再 **减 32** 成有符号 6-bit ✓（范围 −32…31 ✓）；
    * 4 个子块各占 ``qh`` 的 2 个 bit ✓（``k=0/1/2/3`` ⇒ 移位 0/2/4/6 ✓）；
    * 256 元素按 **16 个一段** 取 ``scales[0..15]`` ✓（段号 = 下标 ÷ 16 ✓），
      ``y = d × scales[段] × q`` ✓ —— **只有 scale、没有 min** ✓（Q6_K 不带 ``dmin`` ✓，
      这是与 Q4_K/Q5_K 的又一处不同 ✓）。
    """
    block_bytes = _Q6_K_BLOCK
    if numel % 256 != 0:
        raise GgufDequantError(f"Q6_K 块不整除：numel={numel} ✗")
    nblock = numel // 256
    need = nblock * block_bytes
    if need > len(data):
        raise GgufDequantError(f"Q6_K 数据不够：需要 {need} 字节、只有 {len(data)} ✗")
    raw = np.frombuffer(data, dtype=np.uint8, count=need).reshape(nblock, block_bytes)
    out = np.empty((nblock, 256), dtype=np.float32)
    for start in range(0, nblock, _Q6_K_CHUNK):
        stop = min(start + _Q6_K_CHUNK, nblock)
        chunk = raw[start:stop]
        # ⚠️ 切出来的列**不是 C 连续** ✓✗（行跨度 210 ✓）⇒ view 成 int8/f16 前必须 ascontiguousarray ✓
        scales = np.ascontiguousarray(chunk[:, 192:208]).view(np.int8)
        d = (np.ascontiguousarray(chunk[:, 208:210]).view("<f2")
             .astype(np.float32))                     # (n, 1) ✓
        ql = chunk[:, 0:128]
        qh = chunk[:, 128:192]
        for half in (0, 1):
            ql_h = ql[:, half * 64:half * 64 + 64]
            qh_h = qh[:, half * 32:half * 32 + 32]
            low_h, high_h = ql_h[:, 0:32], ql_h[:, 32:64]
            # 四个子块：低 4 bit 来自 ql 的前/后 32 字节、高 2 bit 来自 qh 的 2k 位 ✓
            packed = np.concatenate([
                (low_h & 0x0F) | (((qh_h >> 0) & 3) << 4),
                (high_h & 0x0F) | (((qh_h >> 2) & 3) << 4),
                (low_h >> 4) | (((qh_h >> 4) & 3) << 4),
                (high_h >> 4) | (((qh_h >> 6) & 3) << 4),
            ], axis=1)                                 # (n, 128) uint8、值域 0…63 ✓
            # 列序 = [q1(32) q2(32) q3(32) q4(32)] ✓ = 块内元素 0…127 的顺序 ✓
            # 段 scale：每 16 列一段、共 8 段 ✓ ⇒ np.repeat(scales_half, 16) 一次对齐全部 128 列 ✓
            # （q2 的 scale 是 sc[2 + l//16] ✓、而列 32+l 的段号 = (32+l)//16 = 2 + l//16 ✓ 自动吻合 ✓）
            scale_cols = np.repeat(scales[:, half * 8:half * 8 + 8].astype(np.float32),
                                   16, axis=1)         # (n, 128) ✓
            # ⚠️ 结合顺序照 llama.cpp（先 ``d * sc`` 再乘 q ✓）—— 换了顺序会差 1 ulp ✗，
            #    「向量化版 == 参考版」的逐位对拍就判不出来 ✗
            out[start:stop, half * 128:(half + 1) * 128] = (
                (packed.view(np.int8) - 32).astype(np.float32) * (d * scale_cols))
    return out.reshape(-1)


_DEQUANT_FN = {0: _dequant_f32, 1: _dequant_f16, 32: _dequant_bf16,
               8: _dequant_q8_0, 12: _dequant_q4_k, 14: _dequant_q6_k}


def dequantize_tensor(data: bytes, type_id: int, numel: int) -> np.ndarray:
    """一块量化字节 ⇒ fp32 **一维** 数组 ✓（未知/未实现的类型 ⇒ **具名拒绝** ✗）。

    ⚠️ 返回的是 **ggml 顺序的一维流** ✓（``idx = i0 + ne0*i1``，``ne0`` 变化最快 ✓），
    **不是**任何 numpy 形状 ✗。要变 2D 必须用 ``reshape(tuple(reversed(dims)))`` ✓
    （见 :func:`gguf_to_llm.numpy_shape` ✓）；按 dims 正序 reshape 会得到「形状对、元素错」✗✗
    —— 2026-10-29 真机 qwen3:8b 全程乱码就是这个 ✗。
    """
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
