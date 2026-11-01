"""comfy 原文的 Wan VAE 参照件 —— **只为生成/复核 golden，不参与产品运行**。

来源（逐字，数学一行未改）：
- ``comfy-studio/ComfyUI/comfy/ldm/wan/vae2_2.py`` 全文，用 ``importlib`` 原样加载；
- 它 ``from .vae import AttentionBlock, CausalConv3d, RMS_norm`` —— 那三个基础件本文件**重抄**
  （``vae.py`` 里它们依赖 ``comfy.ops`` / ``vae_attention``，整包 import 会拖进 ComfyUI 全栈）；
- ``comfy.ldm.modules.diffusionmodules.model.pytorch_attention``（单头 SDPA）同样重抄。

被替换的只有**基础设施**三处，且都是等价物：

===================  ============================================================
``comfy.ops``        ``disable_weight_init`` 的 Conv2d/Conv3d 就是「不初始化权重的
                     nn.Conv2d/3d」，这里直接用 nn 版；``Conv3d.forward`` 多接一个
                     ``autopad`` 关键字，``padding[0]==0`` 时它是 no-op（见下）
``einops.rearrange`` 只实现 vae2_2 用到的 4 个 pattern，语义与 einops 逐字相同
``vae_attention``    等价于 comfy 的 ``pytorch_attention``（只有没装 xformers 时才走它，
                     而 Qwen-Image 2.1 的 VAE 是小模型，官方默认路径就是 SDPA）
===================  ============================================================

⚠️ ``CausalConv3d`` 在 ``t==1 且 cache 为空`` 时走 ``autopad="causal_zero"`` 快路径
（truncating the weight）。本仓关心的 Qwen-Image 2.1 VAE 是 ``temporal_kernel=1``
⇒ 所有 CausalConv3d 的 ``padding[0]`` 都是 0、``_padding`` 恒 0 ⇒ 快路径与普通卷积
逐位相同，所以垫片里让它退化成 ``super().forward(x)`` 是**精确**的，不是近似。

用法::

    from tests.vae_wan_ref import load_reference
    ref = load_reference()                     # 权重从盘上真文件读
    y = ref.decode(z)                          # z: [B, 64, H, W] float32
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

COMFY_ROOT = Path("D:/code/voide/comfy-studio/ComfyUI")
VAE21 = Path("D:/Comfy-Desktop/ComfyUI-Shared/models/vae/qwen_image_2.1_vae_bf16.safetensors")


# ---------------------------------------------------------------- einops 垫片
def rearrange(x: torch.Tensor, pattern: str, **kw) -> torch.Tensor:
    """只覆盖 vae2_2 出现的 4 个 pattern（语义与 einops 一致）。"""
    if pattern == "b c t h w -> (b t) c h w":
        b, c, t, h, w = x.shape
        return x.reshape(b * t, c, h, w)
    if pattern == "(b t) c h w -> b c t h w":
        t = kw["t"]
        bt, c, h, w = x.shape
        return x.reshape(bt // t, t, c, h, w).permute(0, 2, 1, 3, 4)
    if pattern == "b c (h q) (w r) -> b (c r q) h w":
        q = r = kw["q"]
        b, c, hq, wr = x.shape
        return x.reshape(b, c, hq // q, q, wr // r, r).permute(0, 1, 5, 3, 2, 4).reshape(b, c * r * q, hq // q, wr // r)
    if pattern == "b (c r q) h w -> b c (h q) (w r)":
        q = r = kw["q"]
        b, crq, h, w = x.shape
        return x.reshape(b, crq // (r * q), r, q, h, w).permute(0, 1, 4, 3, 5, 2).reshape(b, crq // (r * q), h * q, w * r)
    raise NotImplementedError(f"einops 垫片没实现：{pattern}")


# ------------------------------------------------- 从 comfy/ldm/wan/vae.py 重抄的三个件
class _OpsConv3d(nn.Conv3d):
    """``comfy.ops.disable_weight_init.Conv3d`` 的等价物（不吃 autopad）。"""

    def forward(self, x, autopad=None, **kwargs):
        return super().forward(x)


class CausalConv3d(_OpsConv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = 2 * self.padding[0]
        self.padding = (0, self.padding[1], self.padding[2])

    def forward(self, x, cache_x=None, cache_list=None, cache_idx=None):
        if cache_list is not None:
            cache_x = cache_list[cache_idx]
            cache_list[cache_idx] = None

        if cache_x is None and x.shape[2] == 1:
            return super().forward(x)

        if self._padding > 0:
            padding_needed = self._padding
            if cache_x is not None:
                cache_x = cache_x.to(x.device)
                padding_needed = max(0, padding_needed - cache_x.shape[2])
            padding_shape = list(x.shape)
            padding_shape[2] = padding_needed
            padding = torch.zeros(padding_shape, device=x.device, dtype=x.dtype)
            parts = [p for p in (padding, cache_x, x) if p is not None]
            if len(parts) == 1:
                x = parts[0]
            else:
                x = torch.cat(parts, dim=2)
            del cache_x

        return super().forward(x)


class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else None

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma.to(x) + (self.bias.to(x) if self.bias is not None else 0)


def pytorch_attention(q, k, v):
    """comfy 的 ``pytorch_attention``，逐行照抄。"""
    orig_shape = q.shape
    B = orig_shape[0]
    C = orig_shape[1]
    q, k, v = map(
        lambda t: t.view(B, 1, C, -1).transpose(2, 3).contiguous(),
        (q, k, v),
    )
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
    return out.transpose(2, 3).reshape(orig_shape)


class AttentionBlock(nn.Module):
    """comfy ``wan/vae.py`` 的 AttentionBlock（单头因果自注意力，逐行照抄）。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x, feat_cache=None, feat_idx=[0], final=False):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        x = pytorch_attention(q, k, v)
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        return x + identity


# ------------------------------------------------------------- 伪造 comfy 包并加载原文
def _install_fake_comfy() -> types.ModuleType:
    if "comfy.ldm.wan.vae2_2" in sys.modules:
        return sys.modules["comfy.ldm.wan.vae2_2"]

    einops = types.ModuleType("einops")
    einops.rearrange = rearrange
    sys.modules.setdefault("einops", einops)

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []  # 包
    ops = types.ModuleType("comfy.ops")
    ops.disable_weight_init = types.SimpleNamespace(Conv2d=nn.Conv2d, Conv3d=_OpsConv3d)
    comfy.ops = ops
    sys.modules["comfy"] = comfy
    sys.modules["comfy.ops"] = ops

    for name in ("comfy.ldm", "comfy.ldm.wan"):
        mod = types.ModuleType(name)
        mod.__path__ = []
        sys.modules[name] = mod

    vae = types.ModuleType("comfy.ldm.wan.vae")
    vae.AttentionBlock = AttentionBlock
    vae.CausalConv3d = CausalConv3d
    vae.RMS_norm = RMS_norm
    sys.modules["comfy.ldm.wan.vae"] = vae

    src = COMFY_ROOT / "comfy/ldm/wan/vae2_2.py"
    spec = importlib.util.spec_from_file_location("comfy.ldm.wan.vae2_2", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ 装配 + 解码
def load_reference(weights: Path = VAE21, dtype: torch.dtype = torch.float32):
    """按 ``comfy/sd.py`` 里 Qwen-Image 2.1 那段 ddconfig 装配（参数全部从权重形状读）。"""
    from app.services.engine import safetensors as st
    from app.services.engine.weights import load_module_weights

    mod = _install_fake_comfy()
    info = st.inspect(weights)
    shapes = {k: tuple(e.shape) for k, e in info.tensors.items()}

    model = mod.WanVAE(
        dim=shapes["encoder.conv1.weight"][0],        # 96
        dec_dim=shapes["decoder.head.0.gamma"][0],    # 144
        z_dim=shapes["conv2.weight"][0],              # 64
        dim_mult=[1, 2, 4, 8, 8],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True, True],
        dropout=0.0,
        image_channels=shapes["decoder.head.2.weight"][0],   # 4 (RGBA)
        patch_size=1,
        temporal_kernel=1,
    ).to(dtype)

    report = load_module_weights(model, weights, dtype=dtype)
    if not report.complete:
        raise RuntimeError(f"参考件权重没装全：缺 {report.missing[:5]} 错形 {report.shapeMismatch[:3]} err={report.error}")

    model.eval()
    return model


if __name__ == "__main__":  # 冒烟：真权重解码 8×8 latent
    torch.manual_seed(0)
    ref = load_reference()
    z = torch.randn(1, 64, 8, 8)
    with torch.no_grad():
        y = ref.decode(z)
    print("decode ->", tuple(y.shape), float(y.mean()), float(y.std()))
