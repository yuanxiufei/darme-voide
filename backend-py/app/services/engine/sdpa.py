"""注意力核选择：网络本体的每次 SDPA 都从这里出去。

为什么需要这一层
================
默认 dispatch 在本机（RTX A5000 / sm86 / torch 2.14.1+cu130）选中的是 **EFFICIENT**，
而 **cuDNN fused attention** 在同一形状上快一倍 —— 实测（`data/experiments/nvfp4_dequant_bench/`）：

* 单算子 Qwen-Image 1024²（24 head × 128、seq 4296、bf16、加性 float 掩码）：
  EFFICIENT **6.83 / 7.45 / 3.06 ms** vs CUDNN **3.55 ms**（三例为 1024²／Flux-4608／H3-3460）；
* 真块（`dit_qwen_image._Block` 原样跑，不是我照抄的替身）：**26.6 → 23.4 ms/块（−12.1%）**，
  折算 60 层一步 **1594 → 1402 ms**；峰值分配不变（1.062 GiB，cuDNN workspace 没多吃）；
* 反证 cuDNN **没在偷懒**：`attn_mask_check.py` 里屏蔽 37 个 padding 位后，cuDNN 输出与
  「不传掩码」差 1.1e-1 ⇒ 掩码真被应用；与 float32 显式 softmax 参考的偏差 1.2e-3，
  与 EFFICIENT 同量级（不是「更快所以更不准」）。

⚠️ **只在它自己一个开时才会被选中** —— 本 build 的优先级是 flash → efficient → cudnn → math
（CUDNN 与 EFFICIENT 同时 enable，赢的仍是 EFFICIENT，实测 6.83 ms）⇒ 想用必须
`sdpa_kernel([CUDNN])` 把别的**关掉**，于是「能不能用」必须自己先判，不能靠回退。

代价：成对交出
==============
换核**不是逐位相同**（bf16 归约顺序变）：真块输出 img max|Δ| **3.1e-2** / 均值 6.7e-5、
txt max|Δ| 1.6e-2。也就是说同一 seed 出的图与旧版**不再逐像素一致**。
要逐位复现旧结果 ⇒ `DARME_SDPA=off`。

判据
====
`can_use_cudnn_attention` 在本机 44 个形状上与「强制 CUDNN 真能不能跑」**0 条不符**
（head_dim 256 判 False 且真跑抛；非连续 q 判 True 且真能跑）⇒ 先判后用。
判不过就走原路（行为与改造前**完全一致**），绝不抛异常 —— 把一张图变成一次崩溃不叫优化。

范围
====
只管**网络本体**（DiT / H3 / Music3 等）。文本塔（`clip_text` / `umt5_te` / `qwen35_te`）
**有意不走这里**：它们与 comfy 有 `maxDiff=0` 的逐位对拍钉子，换核会把钉子打掉。
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F

#: ``auto`` ＝ 判得过就用 cuDNN；``off`` ＝ 保持 PyTorch 自己的选择（与改造前逐位相同）
MODE_ENV = "DARME_SDPA"
MODES = ("auto", "off")
DEFAULT_MODE = "auto"

try:                                    # torch ≥ 2.3 才有这两个名字
    from torch.nn.attention import SDPBackend, sdpa_kernel
    _KERNEL_API = True
except ImportError:                     # pragma: no cover - 老 torch 只走原路
    SDPBackend = None                   # type: ignore[assignment]
    sdpa_kernel = None                  # type: ignore[assignment]
    _KERNEL_API = False

_state: dict[str, Any] = {"cudnnCalls": 0, "fallbacks": 0, "unknownMode": None}


def mode() -> str:
    """当前档位：``auto``（默认）/ ``off``。写错的值按 ``auto`` 处理并记进 ``describe()``。"""
    raw = os.environ.get(MODE_ENV)
    if raw is None:
        return DEFAULT_MODE
    value = raw.strip().lower()
    if value in MODES:
        return value
    _state["unknownMode"] = raw
    return DEFAULT_MODE


def set_mode(value: str | None) -> None:
    """测试与（将来的）设置面用的写口；``None`` ＝ 删掉环境变量（回到默认）。"""
    if value is None:
        os.environ.pop(MODE_ENV, None)
        return
    if value not in MODES:
        raise ValueError(f"{MODE_ENV} 只能是 {MODES}，给的是 {value!r}")
    os.environ[MODE_ENV] = value


def describe() -> dict[str, Any]:
    """回执：现在是什么档、这次进程里真走了几层 cuDNN、有没有跌回原路。"""
    return {
        "mode": mode(),
        "cudnnCalls": _state["cudnnCalls"],
        "fallbacks": _state["fallbacks"],
        "unknownMode": _state["unknownMode"],
        "note": "cuDNN 与旧路径**非逐位相同**（bf16 归约顺序不同）；要逐位复现显式设 DARME_SDPA=off",
    }


def reset() -> None:
    """清计数器（测试用）。"""
    _state.update({"cudnnCalls": 0, "fallbacks": 0, "unknownMode": None})


def can_use_cudnn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  attn_mask: torch.Tensor | None = None, *,
                  is_causal: bool = False, enable_gqa: bool = False) -> bool:
    """cuDNN fused attention 在这个形状上**真能不能跑**。

    用 PyTorch 自己的能力判定（`torch.backends.cuda.can_use_cudnn_attention`），
    它读的是与 dispatch 同一份约束 —— 自己写一份 head_dim 阈值才是假绿。
    CPU / 无 CUDA / 老 torch / 判定本身抛异常，一律判 False（走原路）。
    """
    if not _KERNEL_API or not q.is_cuda:
        return False
    backend = getattr(torch.backends, "cuda", None)
    if backend is None or not hasattr(backend, "can_use_cudnn_attention"):
        return False
    try:
        params = backend.SDPAParams(q, k, v, attn_mask, 0.0, is_causal, enable_gqa)
        return bool(backend.can_use_cudnn_attention(params))
    except Exception:  # noqa: BLE001 - 判定不可用不许影响出图
        return False


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              attn_mask: torch.Tensor | None = None, *,
              is_causal: bool = False, enable_gqa: bool = False,
              scale: float | None = None, dropout_p: float = 0.0) -> torch.Tensor:
    """与 ``F.scaled_dot_product_attention`` 同签名；``auto`` 档下优先 cuDNN。

    ``scale`` / ``dropout_p`` 非默认值时**不碰 cuDNN**：那两个口 cuDNN 融合核的
    支持面比 EFFICIENT 窄，不值得为省一次判断去赌（本仓 DiT 都不传这两个）。
    """
    kwargs: dict[str, Any] = {}
    if is_causal:
        kwargs["is_causal"] = True
    if enable_gqa:
        kwargs["enable_gqa"] = True
    if scale is not None:
        kwargs["scale"] = scale
    if dropout_p:
        kwargs["dropout_p"] = dropout_p

    plain = {"attn_mask": attn_mask} if attn_mask is not None else {}
    plain.update(kwargs)

    if (mode() == "auto" and not dropout_p and scale is None
            and can_use_cudnn(q, k, v, attn_mask, is_causal=is_causal, enable_gqa=enable_gqa)):
        try:
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):   # type: ignore[union-attr]
                out = F.scaled_dot_product_attention(q, k, v, **plain)
        except RuntimeError:      # pragma: no cover - 判定说能用却抛 ⇒ 记一笔再走原路
            _state["fallbacks"] += 1
        else:
            _state["cudnnCalls"] += 1
            return out
    return F.scaled_dot_product_attention(q, k, v, **plain)
