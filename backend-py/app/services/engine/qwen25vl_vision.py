"""**Qwen2.5-VL 视觉塔** ✓ —— Qwen-Image **Edit 2511** 的参考图语义编码器 ✓。

## 为什么必须有

Edit 2511 的参考图走**两条**并行通路（`comfy_extras/nodes_qwen.py::TextEncodeQwenImageEditPlus`）：

1. **语义**：参考图经视觉塔 ⇒ ``image_embeds`` ⇒ 拼进文本塔 ``<|image_pad|>`` 位（本模块）；
2. **像素**：参考图经 VAE ⇒ ``ref_latents`` ⇒ 拼进 DiT token 序列（`vae_wan.encode_latents`

缺第 1 条 ⇒ 图对「要改成什么」一个字都不说 ✗（症状：改了但改错地方，**不报错**）。
本仓此前只有 H3 用的 :mod:`.qwen3vl_vision`，那是 **Qwen3-VL** 的，尺子完全不同。

## 口径来源（逐字照抄 `comfy/text_encoders/qwen_vl.py::Qwen2VLVisionTransformer`）

## 盘上实证（`qwen_2.5_vl_7b_fp8_scaled.safetensors` 头，2026-11-01）

```
visual.patch_embed.proj.weight        [1280, 3, 2, 14, 14] BF16  ⇒ hidden 1280 / t 2 / patch 14
visual.blocks.{0..31}.attn.qkv.weight [3840, 1280]         F8_E4M3 ⇒ 3×1280
visual.blocks.{0..31}.mlp.gate_proj   [3420, 1280]         F8_E4M3 ⇒ ffn 3420
visual.merger.mlp.0.weight            [5120, 5120]         F8_E4M3 ⇒ merge 2 ⇒ 1280·4
visual.merger.mlp.2.weight            [3584, 5120]         F8_E4M3 ⇒ 语言塔宽 3584
```

714 张量 = 32 层 × 21 键 + patch_embed 1 + merger 10。线性层是 **fp8_scaled**（每层带
``scale_input`` / ``scale_weight`` 标量）⇒ 必须走 :mod:`.quant` 反量化，当 BF16 直读拿到的是垃圾。

## 五条容易静默错的事实

1. **``num_heads`` 不在权重里**：qkv 合并成 ``[3·hidden, hidden]`` ⇒ 头数只能按档登记（本族 16 ⇒ head_dim 80）。
2. **窗口注意力**：``window_size=112`` ⇒ 合并域窗口 ``112//2//14 = 4``；块 7/15/23/31 走全图。
   重排单位是 **merge 单元**（2×2 个 patch），按 patch 重排形状也对、注意力邻域全错。
3. **旋转位置也按同一套 ``window_index`` 重排**（参考 435-437 行）；只排 hidden ⇒ 每格用邻居角度。
4. **融合后 ``argsort(window_index)`` 还原**（参考 455-457 行，注释 “spatially precise edits”）；
   忘了还原 ⇒ token 顺序与 ``<|image_pad|>`` 位错开（图大致还在、每格内容整体错位）。
5. **qkv 切分是 ``(3, heads, head_dim)`` 交错**；按 ``(heads, 3, head_dim)`` 切形状也过、q/k/v 头全错位。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "FULLATT_BLOCKS", "PATCH_SIZE_25VL", "PIXEL_MEAN_25VL", "PIXEL_STD_25VL", "VISION_25VL",
    "build_vision_tower", "expected_keys", "infer_shape", "patches_of", "pixel_mean_std",
    "shape_problems", "window_plan",
]

#: Qwen2.5-VL-7B 视觉塔尺子 —— ⚠️ ``num_heads`` **只能**来自这张表（权重里没有）
VISION_25VL: dict[str, Any] = {
    "hidden_size": 1280,
    "intermediate_size": 3420,
    "depth": 32,
    "num_heads": 16,
    "in_channels": 3,
    "patch_size": 14,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "window_size": 112,
    "out_hidden_size": 3584,
    "fullatt_block_indexes": (7, 15, 23, 31),
}

PATCH_SIZE_25VL = 14
#: ⚠️ **CLIP 均值方差**（Qwen2.5-VL 口径）；Qwen3-VL 是 0.5/0.5，混用不报错只偏色
PIXEL_MEAN_25VL: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
PIXEL_STD_25VL: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)
#: 走**全图**注意力的层号（其余层走 ``window_size`` 窗口）
FULLATT_BLOCKS: tuple[int, ...] = (7, 15, 23, 31)


def pixel_mean_std() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """像素归一化 (mean, std) —— **CLIP 口径**（照 `qwen_vl.py::QWEN_IMAGE_MEAN/STD`）。"""
    return PIXEL_MEAN_25VL, PIXEL_STD_25VL


def infer_shape(shapes: dict[str, tuple[int, ...]]) -> dict[str, Any]:
    """从**权重形状**反推尺子（不信常量、只信盘上）。只认 ``visual.`` 前缀。

    ⚠️ ``num_heads`` 推不出来（qkv 合并成 ``3·hidden``）⇒ 保留表值，由 :func:`shape_problems` 核自洽。
    """
    shape = dict(VISION_25VL)
    proj = shapes.get("visual.patch_embed.proj.weight")
    if proj is not None and len(proj) == 5:
        shape["hidden_size"] = int(proj[0])
        shape["in_channels"] = int(proj[1])
        shape["temporal_patch_size"] = int(proj[2])
        shape["patch_size"] = int(proj[3])
    gate = shapes.get("visual.blocks.0.mlp.gate_proj.weight")
    if gate is not None and len(gate) == 2:
        shape["intermediate_size"] = int(gate[0])
        shape["hidden_size"] = int(gate[1])
    depth = 0
    while f"visual.blocks.{depth}.attn.qkv.weight" in shapes:
        depth += 1
    if depth:
        shape["depth"] = depth
    qkv = shapes.get("visual.blocks.0.attn.qkv.weight")
    if qkv is not None and len(qkv) == 2:
        #: 只有 ``3·hidden``：头数不在里面 ⇒ 这里**只**核自洽，不反推头数
        shape["qkv_rows"] = int(qkv[0])
    merger = shapes.get("visual.merger.mlp.0.weight")
    if merger is not None and len(merger) == 2:
        hidden = max(int(shape["hidden_size"]), 1)
        unit = int(merger[1]) // hidden
        root = int(round(max(unit, 0) ** 0.5))
        if root in (1, 2, 4, 8) and root * root * hidden == int(merger[1]):
            shape["spatial_merge_size"] = root
    out = shapes.get("visual.merger.mlp.2.weight")
    if out is not None and len(out) == 2:
        shape["out_hidden_size"] = int(out[0])
    return shape


def shape_problems(shape: dict[str, Any]) -> list[str]:
    """尺子自洽检查 —— 空列表才算过。"""
    problems: list[str] = []
    hidden = int(shape["hidden_size"])
    heads = int(shape["num_heads"])
    if heads <= 0 or hidden % heads:
        problems.append(f"hidden {hidden} 不能被 num_heads {heads} 整除（head_dim 是小数）")
    elif (hidden // heads) % 4:
        problems.append(f"head_dim {hidden // heads} 不能被 4 整除（旋转要成对切半，本族是 80）")
    qkv_rows = shape.get("qkv_rows")
    if qkv_rows is not None and int(qkv_rows) != 3 * hidden:
        problems.append(f"qkv 行数 {qkv_rows} ≠ 3×hidden {3 * hidden}（合并口径变了）")
    fullatt = tuple(int(i) for i in shape.get("fullatt_block_indexes", ()))
    depth = int(shape["depth"])
    bad = [i for i in fullatt if not 0 <= i < depth]
    if bad:
        problems.append(f"全图注意力层号 {bad} 越界（depth {depth}）")
    return problems


def expected_keys(shape: dict[str, Any]) -> set[str]:
    """**应占**的权重键（不含 ``.scale_input`` / ``.scale_weight`` 这两个量化伴随件）。

    「少接一层」是最典型的静默错：多出来的键会被丢掉而不报错 ⇒ 必须逐键对账。
    """
    keys = {"visual.patch_embed.proj.weight", "visual.merger.ln_q.weight",
            "visual.merger.mlp.0.weight", "visual.merger.mlp.0.bias",
            "visual.merger.mlp.2.weight", "visual.merger.mlp.2.bias"}
    for index in range(int(shape["depth"])):
        base = f"visual.blocks.{index}"
        keys |= {
            f"{base}.norm1.weight", f"{base}.norm2.weight",
            f"{base}.attn.qkv.weight", f"{base}.attn.qkv.bias",
            f"{base}.attn.proj.weight", f"{base}.attn.proj.bias",
            f"{base}.mlp.gate_proj.weight", f"{base}.mlp.gate_proj.bias",
            f"{base}.mlp.up_proj.weight", f"{base}.mlp.up_proj.bias",
            f"{base}.mlp.down_proj.weight", f"{base}.mlp.down_proj.bias",
        }
    return keys


def window_plan(grid_thw: Any, shape: dict[str, Any]) -> tuple[Any, list[int]]:
    """算 ``(window_index, cu_window_seqlens)`` —— 逐字照参考 ``get_window_index``。

    ``grid_thw``: 逐图 ``[t, h, w]``（patch 单位）。返回的重排下标是**merge 单元**上的。
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as functional  # noqa: PLC0415

    merge = int(shape["spatial_merge_size"])
    patch = int(shape["patch_size"])
    #: ⚠️ 窗口是**合并域**单位：112 // merge // patch = 4
    window = int(shape["window_size"]) // merge // patch
    if window <= 0:
        raise ValueError(f"窗口 {shape['window_size']} 在合并域不足一格（merge {merge} / patch {patch}）")
    index_parts: list[Any] = []
    cu: list[int] = [0]
    offset = 0
    for frames, height, width in grid_thw.tolist():
        frames, height, width = int(frames), int(height), int(width)
        if height % merge or width % merge:
            raise ValueError(f"网格 {height}×{width} 不能被 merge {merge} 整除（预处理该切齐）")
        rows, cols = height // merge, width // merge
        idx = torch.arange(frames * rows * cols).reshape(frames, rows, cols)
        pad_h = window - rows % window
        pad_w = window - cols % window
        windows_h = (rows + pad_h) // window
        windows_w = (cols + pad_w) // window
        padded = functional.pad(idx, (0, pad_w, 0, pad_h), "constant", -100)
        padded = padded.reshape(frames, windows_h, window, windows_w, window)
        padded = padded.permute(0, 1, 3, 2, 4).reshape(
            frames, windows_h * windows_w, window, window)
        seqlens = (padded != -100).sum([2, 3]).reshape(-1)
        flat = padded.reshape(-1)
        index_parts.append(flat[flat != -100] + offset)
        cu.extend((seqlens.cumsum(0) * merge * merge + cu[-1]).tolist())
        offset += frames * rows * cols
    #: ⚠️ 去重相邻（空窗口会重复出同一个累计值）—— 参考在调 forward 前做 ``unique_consecutive``
    deduped: list[int] = [cu[0]]
    for value in cu[1:]:
        if value != deduped[-1]:
            deduped.append(int(value))
    return torch.cat(index_parts, dim=0), deduped


def patches_of(pixels: Any, shape: dict[str, Any] | None = None) -> tuple[Any, tuple[int, int, int]]:
    """单图 ``(H, W, C)`` ⇒ ``(patches, (1, grid_h, grid_w))``，**本族口径**（patch 14 + CLIP 均值）。

    ⚠️ 归一化与网格必须**同一套**：这里把 ``patch_size`` / ``mean`` / ``std`` 一起交给
    :func:`.qwen3vl_pixels.process_image_with` ✓ —— 那正是「参考图偏色、网格还被按 16 算」的来源 ✓✗。
    """
    from .qwen3vl_pixels import process_image_with  # noqa: PLC0415

    plan = dict(VISION_25VL if shape is None else shape)
    return process_image_with(
        pixels,
        mean_values=PIXEL_MEAN_25VL,
        std_values=PIXEL_STD_25VL,
        patch_size=int(plan["patch_size"]),
        merge_size=int(plan["spatial_merge_size"]),
    )


def _torch_parts() -> dict[str, Any]:
    """把 ``torch`` 依赖**推迟到调用期**（`capabilities()` 契约＝不 import torch、毫秒级）。"""
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415
    functional = torch.nn.functional

    class RMSNorm(nn.Module):
        """**RMSNorm**（``eps=1e-6``）—— 权重只有 ``.weight``、没有 bias。

        当 LayerNorm 用 ⇒ 均值被一起减掉 ⇒ 视觉 token 整体偏移（形状全对、不报错）。
        """

        def __init__(self, dim: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = float(eps)

        def forward(self, x: Any) -> Any:
            work = x.float()
            out = work * torch.rsqrt(work.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return (out * self.weight.float()).to(x.dtype)

    class PatchEmbed(nn.Module):
        """``(tokens, C·t·p·p)`` ⇒ ``(tokens, hidden)``（Conv3d 核 = 步长）。"""

        def __init__(self, shape: dict[str, Any]) -> None:
            super().__init__()
            kernel = (int(shape["temporal_patch_size"]), int(shape["patch_size"]),
                      int(shape["patch_size"]))
            self.proj = nn.Conv3d(int(shape["in_channels"]), int(shape["hidden_size"]),
                                  kernel_size=kernel, stride=kernel, bias=False)
            self.t = kernel[0]
            self.p = kernel[1]
            self.channels = int(shape["in_channels"])
            self.hidden = int(shape["hidden_size"])

        def forward(self, patches: Any) -> Any:
            rows = int(patches.shape[0])
            x = patches.reshape(rows, self.channels, self.t, self.p, self.p)
            #: ⚠️ 输入精度**跟着权重走**（像素是 f32 算的 ✓、塔可能是 bf16 ✓）——
            #:    不在这里对齐，报错会指向 `F.conv3d`「Input type FloatTensor vs weight BFloat16」，
            #:    读起来像「权重坏了」而其实是像素没降精度（与 `dit_lumina` 那三处同型 ✓）。
            return self.proj(x.to(self.proj.weight.dtype)).reshape(rows, self.hidden)

    class VisionMLP(nn.Module):
        """SwiGLU（``silu(gate) * up`` ⇒ down）—— 用 relu/gelu 不报错、只是数值不对。"""

        def __init__(self, hidden: int, intermediate: int) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(hidden, intermediate, bias=True)
            self.up_proj = nn.Linear(hidden, intermediate, bias=True)
            self.down_proj = nn.Linear(intermediate, hidden, bias=True)

        def forward(self, x: Any) -> Any:
            return self.down_proj(functional.silu(self.gate_proj(x)) * self.up_proj(x))

    class Attention(nn.Module):
        def __init__(self, hidden: int, heads: int) -> None:
            super().__init__()
            self.heads = int(heads)
            self.head_dim = hidden // int(heads)
            self.qkv = nn.Linear(hidden, hidden * 3, bias=True)
            self.proj = nn.Linear(hidden, hidden, bias=True)

        @staticmethod
        def _rotate_half(x: Any) -> Any:
            half = x.shape[-1] // 2
            return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

        def forward(self, x: Any, cos: Any, sin: Any, lengths: list[int]) -> Any:
            """``cos``/``sin``: ``(tokens, head_dim)``；``lengths``: 逐段长度（段间不许互看）。"""
            seq = int(x.shape[0])
            qkv = self.qkv(x)
            #: ⚠️ ``(seq, 3, heads, head_dim)`` —— 写成 ``(seq, heads, 3, head_dim)`` 形状也过、头全错位
            q, k, v = qkv.reshape(seq, 3, self.heads, self.head_dim).permute(1, 0, 2, 3).unbind(0)
            qf, kf = q.float(), k.float()
            cosf, sinf = cos.float().unsqueeze(-2), sin.float().unsqueeze(-2)
            qf = (qf * cosf) + (self._rotate_half(qf) * sinf)
            kf = (kf * cosf) + (self._rotate_half(kf) * sinf)
            scale = self.head_dim ** -0.5
            pieces: list[Any] = []
            offset = 0
            for length in lengths:
                end = offset + int(length)
                if end > seq:
                    raise ValueError(
                        f"注意力分段越界（{end} > {seq}）—— 窗口段表与窗口重排不同源，不许补零")
                qi = qf[offset:end].transpose(0, 1).unsqueeze(0)
                ki = kf[offset:end].transpose(0, 1).unsqueeze(0)
                vi = v[offset:end].float().transpose(0, 1).unsqueeze(0)
                pieces.append(functional.scaled_dot_product_attention(
                    qi, ki, vi, scale=scale).squeeze(0).transpose(0, 1))
                offset = end
            if offset != seq:
                raise ValueError(f"注意力分段只盖了 {offset} 个 token、实际 {seq} 个")
            merged = torch.cat(pieces, dim=0) if pieces else x.new_zeros((seq, self.heads,
                                                                          self.head_dim))
            #: ⚠️ 分段结果要按 ``(seq, heads·head_dim)`` 拼回 —— 少这一句就是
            #:    ``(seq, heads, head_dim)`` 直接喂 proj（形状对不上，报错 ✓ 这条不静默 ✓）
            return self.proj(merged.reshape(seq, -1).to(x.dtype))

    class VisionBlock(nn.Module):
        def __init__(self, hidden: int, intermediate: int, heads: int) -> None:
            super().__init__()
            self.norm1 = RMSNorm(hidden)
            self.norm2 = RMSNorm(hidden)
            self.attn = Attention(hidden, heads)
            self.mlp = VisionMLP(hidden, intermediate)

        def forward(self, x: Any, cos: Any, sin: Any, lengths: list[int]) -> Any:
            x = x + self.attn(self.norm1(x), cos, sin, lengths)
            return x + self.mlp(self.norm2(x))

    class PatchMerger(nn.Module):
        """``merge²`` 个 patch ⇒ ``out_hidden``（``ln_q`` 在 **view 之前** = pre-shuffle）。"""

        def __init__(self, hidden: int, merge: int, out_hidden: int) -> None:
            super().__init__()
            self.width = hidden * merge * merge
            self.ln_q = RMSNorm(hidden)
            self.mlp = nn.Sequential(
                nn.Linear(self.width, self.width),
                nn.GELU(),
                nn.Linear(self.width, out_hidden),
            )

        def forward(self, x: Any) -> Any:
            return self.mlp(self.ln_q(x).reshape(-1, self.width))

    class VisionTower(nn.Module):
        """Qwen2.5-VL 视觉塔：``forward(patches, grid_thw)`` ⇒ ``(tokens/merge², out_hidden)``。"""

        def __init__(self, shape: dict[str, Any]) -> None:
            super().__init__()
            hidden = int(shape["hidden_size"])
            heads = int(shape["num_heads"])
            self.hidden_size = hidden
            self.heads = heads
            self.head_dim = hidden // heads
            self.merge = int(shape["spatial_merge_size"])
            self.shape = dict(shape)
            self.patch_embed = PatchEmbed(shape)
            self.blocks = nn.ModuleList([
                VisionBlock(hidden, int(shape["intermediate_size"]), heads)
                for _ in range(int(shape["depth"]))
            ])
            self.merger = PatchMerger(hidden, self.merge, int(shape["out_hidden_size"]))
            self.fullatt = tuple(int(i) for i in shape["fullatt_block_indexes"])
            #: ⚠️ 40 个频率 ⇒ 出 80 宽（2D 旋转，行/列两轴各 40）⇒ ``cat`` 后 80 = head_dim ✓
            inv = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim // 2, 2,
                                                  dtype=torch.float) / (self.head_dim // 2)))
            self.register_buffer("inv_freq", inv, persistent=False)

        def positions(self, grid_thw: Any) -> Any:
            """逐图 ``(t,h,w)`` ⇒ ``(tokens, head_dim/2)`` 的 2D 旋转角（**未**做窗口重排）。"""
            width_half = self.head_dim // 2
            parts: list[Any] = []
            device = self.inv_freq.device
            for frames, height, width in grid_thw.tolist():
                frames, height, width = int(frames), int(height), int(width)
                rows = torch.arange(height, device=device).unsqueeze(1).expand(-1, width)
                rows = rows.reshape(height // self.merge, self.merge,
                                    width // self.merge, self.merge)
                rows = rows.permute(0, 2, 1, 3).flatten()
                cols = torch.arange(width, device=device).unsqueeze(0).expand(height, -1)
                cols = cols.reshape(height // self.merge, self.merge,
                                    width // self.merge, self.merge)
                cols = cols.permute(0, 2, 1, 3).flatten()
                pairs = torch.stack([rows, cols], dim=-1).repeat(frames, 1)
                table = torch.outer(torch.arange(max(height, width), device=device).float(),
                                    self.inv_freq)
                parts.append(table[pairs].flatten(1))
            rotary = torch.cat(parts, dim=0)
            if rotary.shape[-1] != width_half:
                raise ValueError(
                    f"旋转宽度 {rotary.shape[-1]} ≠ head_dim//2 {width_half}"
                    f"（head_dim {self.head_dim} 必须能被 4 整除）")
            return rotary

        def forward(self, patches: Any, grid_thw: Any) -> Any:
            """``patches``: ``(tokens, C·t·p·p)``；``grid_thw``: ``(n, 3)`` patch 单位。"""
            expected = int(sum(int(t) * int(h) * int(w) for t, h, w in grid_thw.tolist()))
            if expected != int(patches.shape[0]):
                raise ValueError(
                    f"网格算出 {expected} 个 patch、实际给了 {int(patches.shape[0])} 个"
                    f"（grid_thw 与像素不同源 —— 不许猜、也不许补零）")
            hidden = self.patch_embed(patches)
            index, cu_window = window_plan(grid_thw, self.shape)
            unit = self.merge * self.merge
            seq = int(hidden.shape[0])
            #: 窗口重排作用在 **merge 单元** 上（4 个 patch 一组），hidden 与位置表**同一套**
            hidden = hidden.reshape(seq // unit, unit, -1)[index, :, :].reshape(seq, -1)
            pos = self.positions(grid_thw).reshape(seq // unit, unit, -1)[index, :, :]
            pos = torch.cat((pos.reshape(seq, -1),) * 2, dim=-1)
            cos, sin = pos.cos(), pos.sin()
            fullatt: list[int] = []
            for frames, height, width in grid_thw.tolist():
                fullatt.extend([int(height) * int(width)] * int(frames))
            window_lengths = [b - a for a, b in zip(cu_window, cu_window[1:])]
            for position, block in enumerate(self.blocks):
                lengths = fullatt if position in self.fullatt else window_lengths
                hidden = block(hidden, cos, sin, lengths)
            merged = self.merger(hidden)
            #: ⚠️ 还原窗口顺序（参考 455-457 行）—— 不还原 ⇒ 与 ``<|image_pad|>`` 位错开
            return merged[torch.argsort(index)]

    def build_vision_tower(shape: dict[str, Any], dtype: Any = None, device: Any = None) -> Any:
        """造一台视觉塔。``device="meta"`` ⇒ 在 meta 上构造（不白分配实存）。"""
        if device is not None and str(device) == "meta":
            with torch.device("meta"):
                tower = VisionTower(shape)
        else:
            tower = VisionTower(shape)
        tower.shape = dict(shape)
        if dtype is not None or device is not None:
            tower = tower.to(device=device, dtype=dtype)
        return tower

    return {"torch": torch, "nn": nn, "VisionTower": VisionTower,
            "build_vision_tower": build_vision_tower}


def build_vision_tower(shape: dict[str, Any], dtype: Any = None, device: Any = None) -> Any:
    """模块级入口（``_torch_parts()`` 里有同名实现，这里只转一手）。"""
    return _torch_parts()["build_vision_tower"](shape, dtype, device)
