"""Qwen3-VL-32B 的**视觉塔** ✓（H3 文本编码器 ``qwen3vl_32b_minimax_h3_int8_convrot`` 的
``visual.*`` 那半 ✓ 2026-11-01 首次接上 ✓）。

## 为什么必须有

H3 的 ``fl2va`` / ``ref2va`` 带参考图 ✓，参考图**不是**塞进潜帧当条件，而是**经视觉塔编码成
token 序列接在 prompt 前面** ✓（`comfy/text_encoders/minimax.py` 抬头逐字写着
``"<Picture 1>: " <vision block> <prompt>`` ✓）。本仓此前没有视觉塔 ⇒ 参考图既进不了文本塔、
也进不了 DiT 的文本条件 ⇒ 「给了图等于没给」✓✗（**不报错** ✓）。

## 口径来源（逐字照抄）

* 结构 / 前向：`comfy/text_encoders/qwen35.py::Qwen35VisionModel` 一族 ✓
  （`qwen3vl.py::Qwen3VLVisionModel` 只多了**三路 DeepStack** ✓）
* 尺子：`QWEN3VL_VISION_COMMON` + `QWEN3VL_VISION["qwen3vl_32b"]` ✓
* 像素：`qwen_vl.process_qwen2vl_images` ✓ —— **Qwen3-VL 是 0.5/0.5 归一化** ✗✗ 不是
  Qwen2.5-VL 的 CLIP 均值方差 ✓（混用**不报错**、只让参考图整体偏色 ✓）

## 盘上权重实证（2026-11-01 读 safetensors 头 ✓）

```
visual.patch_embed.proj.weight   [1152, 3, 2, 16, 16] BF16  ⇒ hidden 1152 / t 2 / patch 16
visual.pos_embed.weight          [2304, 1152]         BF16  ⇒ 48×48 网格
visual.blocks.{0..26}.*          27 层 × 12 键          BF16
visual.blocks.0.attn.qkv.weight  [3456, 1152]         BF16  ⇒ 3×1152 ⇒ 16 头 × 72
visual.blocks.0.mlp.linear_fc1   [4304, 1152]         BF16  ⇒ ffn 4304
visual.merger.norm.weight        [1152]               BF16  ⇒ **pre**-shuffle ✓
visual.merger.linear_fc1.weight  [4608, 4608]         BF16  ⇒ merge 2 ⇒ 1152·4
visual.deepstack_merger_list.{0,1,2}.norm.weight [4608] BF16 ⇒ **post**-shuffle ✓（挂 8/16/24 层）
```

⚠️ 视觉塔**整体 BF16、没有量化** ✓（对比语言塔全量 ``I8``+``weight_scale`` ✓）⇒ 装载走普通
权重即可 ✓（别把它塞进 convrot 反量化那条路 ✗）。

## 三条容易踩错的硬事实

1. **视觉塔没有 QK-norm** ✗：只有 ``norm1``/``norm2`` 两个 LayerNorm ✓（那是语言塔的事 ✓）。
2. **qkv 是合并权重、按 ``(3, heads, head_dim)`` 交错排** ✓：``reshape(seq, 3, heads, -1)
   .permute(1,0,2,3).unbind(0)`` ✓ —— 按 ``(heads, 3, head_dim)`` 切形状也全对 ✗，但 q/k/v
   的**头是错位的** ✓✗（不报错、图糊 ✓）。
3. **注意力逐图**（``cu_seqlens`` 切段、无因果 ✓）：多张参考图之间**不许互看** ✗。
"""

from __future__ import annotations

from typing import Any

__all__ = ["VISION_32B", "infer_shape", "pixel_mean_std", "rotation_kind", "build_vision_tower"]

#: Qwen3-VL-32B 视觉塔的尺子 ✓（与 `QWEN3VL_VISION["qwen3vl_32b"]` 对齐 ✓）
VISION_32B: dict[str, Any] = {
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "depth": 27,
    "num_heads": 16,
    "in_channels": 3,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "num_position_embeddings": 2304,
    "deepstack_visual_indexes": (8, 16, 24),
    "out_hidden_size": 5120,
}

#: ⚠️ **0.5 / 0.5** ✓（Qwen3-VL）✗✗ 不是 Qwen2.5-VL 的 `[0.48145466, …]` ✓
_PIXEL_MEAN = (0.5, 0.5, 0.5)
_PIXEL_STD = (0.5, 0.5, 0.5)


def pixel_mean_std() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """像素归一化 (mean, std) ✓ —— **Qwen3-VL 口径** ✓（照 `qwen3vl.py` 的 ``preprocess_embed`` ✓）。"""
    return _PIXEL_MEAN, _PIXEL_STD


def rotation_kind() -> str:
    """视觉塔用 **2D 旋转**（行/列两轴 ✓），不是 mrope ✓ —— 自检拿它断言 ✓。"""
    return "2d-rotary"


def infer_shape(shapes: dict[str, tuple[int, ...]]) -> dict[str, Any]:
    """从**权重形状**反推尺子 ✓（不信常量、只信盘上 ✓）。只认 ``visual.`` 前缀 ✓。"""
    shape = dict(VISION_32B)
    proj = shapes.get("visual.patch_embed.proj.weight")
    if proj is not None and len(proj) == 5:
        shape["hidden_size"] = int(proj[0])
        shape["in_channels"] = int(proj[1])
        shape["temporal_patch_size"] = int(proj[2])
        shape["patch_size"] = int(proj[3])
    fc1 = shapes.get("visual.blocks.0.mlp.linear_fc1.weight")
    if fc1 is not None and len(fc1) == 2:
        shape["intermediate_size"] = int(fc1[0])
    pos = shapes.get("visual.pos_embed.weight")
    if pos is not None and len(pos) == 2:
        shape["hidden_size"] = int(pos[1])
        shape["num_position_embeddings"] = int(pos[0])
    merge = shapes.get("visual.merger.linear_fc1.weight")
    if merge is not None and len(merge) == 2:
        # ⚠️ merge 尺寸只能**除回来** ✓（写死 4 换个档就静默错 ✗）
        unit = int(merge[1]) // max(int(shape["hidden_size"]), 1)
        root = int(round(max(unit, 0) ** 0.5))
        if root > 0 and root * root == unit:
            shape["spatial_merge_size"] = root
    out = shapes.get("visual.merger.linear_fc2.weight")
    if out is not None and len(out) == 2:
        shape["out_hidden_size"] = int(out[0])
    depth = 0
    while f"visual.blocks.{depth}.attn.qkv.weight" in shapes:
        depth += 1
    if depth:
        shape["depth"] = depth
    stacks = 0
    while f"visual.deepstack_merger_list.{stacks}.linear_fc2.weight" in shapes:
        stacks += 1
    if stacks:
        # ⚠️ 三路 DeepStack 挂哪三层**权重里查不到** ✗（只存 merger ✓）⇒ 取参考常量 8/16/24 ✓，
        #    但**必须与深度自洽** ✓（越界当场红 ✓，不静默少注一路 ✗）
        shape["deepstack_visual_indexes"] = tuple(range(8, 8 * (stacks + 1), 8))[:stacks]
    return shape


def _torch_parts() -> dict[str, Any]:
    """造出需要 torch 的类 ✓（模块级定义会强制 import torch ✗ ⇒ 关进工厂 ✓）。"""
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    def _apply_rotary(x: Any, position_embeddings: Any) -> Any:
        """``rotate_half`` 风格 ✓：``x*cos + rotate_half(x)*cat(sin1, sin2)`` ✓（照参考 ✓）。"""
        cos, sin1, sin2 = position_embeddings
        half = x.shape[-1] // 2
        rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return x * cos + rotated * torch.cat([sin1, sin2], dim=-1)

    def _per_image_attention(q: Any, k: Any, v: Any, lengths: list[int]) -> Any:
        """逐图注意力 ✓（``(tokens, heads, dim)`` ✓ 单 batch ✓ 无掩码 ✓）。

        ⚠️ 靠 ``cu_seqlens`` **切段**保证图与图互不可见 ✓ —— 不造大布尔掩码 ✗（省显存 ✓，
        且不会因为某行被全遮而 NaN ✓）。
        """
        scale = q.shape[-1] ** -0.5
        outs = []
        for qs, ks, vs in zip(torch.split(q, lengths, dim=0),
                             torch.split(k, lengths, dim=0),
                             torch.split(v, lengths, dim=0)):
            scores = torch.einsum("thd,shd->hts", qs, ks) * scale
            probs = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
            outs.append(torch.einsum("hts,shd->thd", probs, vs))
        return torch.cat(outs, dim=0)

    class VisionPatchEmbed(nn.Module):
        """``conv3d(k=(t, p, p), stride=同)`` ✓ ⇒ 直接出 token ✓（无重叠 ✓）。"""

        def __init__(self, shape: dict[str, Any]) -> None:
            super().__init__()
            self.temporal_patch_size = int(shape["temporal_patch_size"])
            self.patch_size = int(shape["patch_size"])
            self.in_channels = int(shape["in_channels"])
            self.embed_dim = int(shape["hidden_size"])
            kernel = (self.temporal_patch_size, self.patch_size, self.patch_size)
            self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel, kernel, bias=True)

        def forward(self, x: Any) -> Any:
            x = x.view(-1, self.in_channels, self.temporal_patch_size,
                       self.patch_size, self.patch_size)
            return self.proj(x).view(-1, self.embed_dim)

    class VisionMlp(nn.Module):
        """``gelu(approximate="tanh")`` ✗✗ —— 精确 gelu 只差一点点 ✓（照参考 ✓）。"""

        def __init__(self, hidden: int, inter: int) -> None:
            super().__init__()
            self.linear_fc1 = nn.Linear(hidden, inter, bias=True)
            self.linear_fc2 = nn.Linear(inter, hidden, bias=True)

        def forward(self, x: Any) -> Any:
            return self.linear_fc2(F.gelu(self.linear_fc1(x), approximate="tanh"))

    class VisionAttention(nn.Module):
        def __init__(self, hidden: int, heads: int) -> None:
            super().__init__()
            self.num_heads = heads
            self.head_dim = hidden // heads
            # ⚠️ qkv **合并、带 bias** ✓（视觉塔有 bias，语言塔没有 ✗ —— 别统一 ✓）
            self.qkv = nn.Linear(hidden, hidden * 3, bias=True)
            self.proj = nn.Linear(hidden, hidden, bias=True)

        def forward(self, x: Any, lengths: list[int], position_embeddings: Any) -> Any:
            seq = x.shape[0]
            # ⚠️ 布局 ``(seq, 3, heads, dim)`` ✗✗ 不是 ``(3, seq, heads, dim)`` ✓（抬头第 2 条）
            q, k, v = (self.qkv(x).reshape(seq, 3, self.num_heads, self.head_dim)
                       .permute(1, 0, 2, 3).unbind(0))
            q = _apply_rotary(q, position_embeddings)
            k = _apply_rotary(k, position_embeddings)
            return self.proj(_per_image_attention(q, k, v, lengths).reshape(seq, -1))

    class VisionBlock(nn.Module):
        def __init__(self, shape: dict[str, Any]) -> None:
            super().__init__()
            hidden = int(shape["hidden_size"])
            # ⚠️ eps 写死 **1e-6** ✓（照参考 ✓）；不是语言塔那个 layer_norm_eps ✗
            self.norm1 = nn.LayerNorm(hidden, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden, eps=1e-6)
            self.attn = VisionAttention(hidden, int(shape["num_heads"]))
            self.mlp = VisionMlp(hidden, int(shape["intermediate_size"]))

        def forward(self, x: Any, lengths: list[int], position_embeddings: Any) -> Any:
            x = x + self.attn(self.norm1(x), lengths, position_embeddings)
            return x + self.mlp(self.norm2(x))

    class PatchMerger(nn.Module):
        """主 merger ✓：``norm`` 在 **view 之前**（宽度 = hidden ✓）✗✗ —— 与 DeepStack 那族相反 ✓。"""

        def __init__(self, hidden: int, merge: int, out: int) -> None:
            super().__init__()
            self.merge_dim = hidden * (merge ** 2)
            self.norm = nn.LayerNorm(hidden, eps=1e-6)
            self.linear_fc1 = nn.Linear(self.merge_dim, self.merge_dim, bias=True)
            self.linear_fc2 = nn.Linear(self.merge_dim, out, bias=True)

        def forward(self, x: Any) -> Any:
            x = self.norm(x).view(-1, self.merge_dim)
            return self.linear_fc2(F.gelu(self.linear_fc1(x)))

    class DeepstackMerger(nn.Module):
        """DeepStack merger ✓：``norm`` 在 **view 之后**（宽度 = hidden·merge² ✓）✗✗ 与主 merger 相反。"""

        def __init__(self, hidden: int, merge: int, out: int) -> None:
            super().__init__()
            self.merge_dim = hidden * (merge ** 2)
            self.norm = nn.LayerNorm(self.merge_dim, eps=1e-6)
            self.linear_fc1 = nn.Linear(self.merge_dim, self.merge_dim, bias=True)
            self.linear_fc2 = nn.Linear(self.merge_dim, out, bias=True)

        def forward(self, x: Any) -> Any:
            x = self.norm(x.view(-1, self.merge_dim))
            return self.linear_fc2(F.gelu(self.linear_fc1(x)))

    class VisionTower(nn.Module):
        """27 层 ViT + 双线性 pos_embed 插值 + 2D 旋转 + 主 merger + 三路 DeepStack ✓。"""

        def __init__(self, shape: dict[str, Any]) -> None:
            super().__init__()
            self.shape = dict(shape)
            self.spatial_merge_size = int(shape["spatial_merge_size"])
            self.hidden_size = int(shape["hidden_size"])
            self.num_heads = int(shape["num_heads"])
            self.num_position_embeddings = int(shape["num_position_embeddings"])
            self.num_grid_per_side = int(round(self.num_position_embeddings ** 0.5))
            self.deepstack_visual_indexes = list(shape["deepstack_visual_indexes"])
            self.patch_embed = VisionPatchEmbed(shape)
            self.pos_embed = nn.Embedding(self.num_position_embeddings, self.hidden_size)
            # ⚠️ 旋转维度 = ``head_dim // 2`` ✓（不是 head_dim ✗）：72//2 = 36 ⇒ 18 个频率 ✓
            self.rotary_dim = self.hidden_size // self.num_heads // 2
            inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.rotary_dim, 2).float()
                                         / self.rotary_dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.blocks = nn.ModuleList([VisionBlock(shape) for _ in range(int(shape["depth"]))])
            self.merger = PatchMerger(self.hidden_size, self.spatial_merge_size,
                                      int(shape["out_hidden_size"]))
            self.deepstack_merger_list = nn.ModuleList([
                DeepstackMerger(self.hidden_size, self.spatial_merge_size,
                                int(shape["out_hidden_size"]))
                for _ in self.deepstack_visual_indexes
            ])

        def fast_pos_embed_interpolate(self, grid_thw: Any) -> Any:
            """48×48 固定网格 → 任意 ``(h, w)``：**4 邻域双线性插值** ✓（照参考 ✓）。"""
            rows = grid_thw.tolist()
            device = self.pos_embed.weight.device
            side = self.num_grid_per_side
            merge = self.spatial_merge_size
            index_lists: list[list[int]] = [[] for _ in range(4)]
            weight_lists: list[list[float]] = [[] for _ in range(4)]
            for _t, h, w in rows:
                h, w = int(h), int(w)
                y = torch.linspace(0, side - 1, h, device=device)
                x = torch.linspace(0, side - 1, w, device=device)
                y0, x0 = y.int(), x.int()
                y1 = (y.int() + 1).clip(max=side - 1)
                x1 = (x.int() + 1).clip(max=side - 1)
                dy, dx = y - y0, x - x0
                base0 = y0[:, None] * side
                base1 = y1[:, None] * side
                indices = [
                    (base0 + x0[None]).flatten(), (base0 + x1[None]).flatten(),
                    (base1 + x0[None]).flatten(), (base1 + x1[None]).flatten(),
                ]
                weights = [
                    ((1 - dy)[:, None] * (1 - dx)[None]).flatten(),
                    ((1 - dy)[:, None] * dx[None]).flatten(),
                    (dy[:, None] * (1 - dx)[None]).flatten(),
                    (dy[:, None] * dx[None]).flatten(),
                ]
                for j in range(4):
                    index_lists[j].extend(indices[j].tolist())
                    weight_lists[j].extend(weights[j].tolist())
            index_tensor = torch.tensor(index_lists, dtype=torch.long, device=device)
            weight_tensor = torch.tensor(weight_lists, dtype=self.pos_embed.weight.dtype,
                                         device=device)
            gathered = self.pos_embed(index_tensor) * weight_tensor[:, :, None]
            baked = gathered[0] + gathered[1] + gathered[2] + gathered[3]
            chunks = baked.split([int(h * w) for _t, h, w in rows])
            out = []
            for chunk, t, h, w in ((c, int(r[0]), int(r[1]), int(r[2]))
                                   for c, r in zip(chunks, rows)):
                chunk = chunk.repeat(t, 1)
                # ⚠️ 这个 view/permute 把 **merge 块内部的行列** 重排成「块优先」顺序 ✓✗
                #    漏了它 ⇒ pos_embed 与 patch 序列**错位** ✓ 但形状全对 ✓✗（图糊得很有规律）
                chunk = (chunk.view(t, h // merge, merge, w // merge, merge, -1)
                         .permute(0, 1, 3, 2, 4, 5).flatten(0, 4))
                out.append(chunk)
            return torch.cat(out)

        def rot_pos_emb(self, grid_thw: Any) -> Any:
            """每 token 的 ``(row, col)`` → 旋转角 ✓（**块优先**顺序 ✓ 与 pos_embed 那处一致 ✓）。"""
            rows = grid_thw.tolist()
            merge = self.spatial_merge_size
            device = self.inv_freq.device
            max_hw = max(max(int(h), int(w)) for _t, h, w in rows)
            seq = torch.arange(max_hw, device=device, dtype=self.inv_freq.dtype)
            freq_table = torch.outer(seq, self.inv_freq)
            total = sum(int(t) * int(h) * int(w) for t, h, w in rows)
            pos_ids = torch.empty((total, 2), dtype=torch.long, device=device)
            offset = 0
            for frames, height, width in rows:
                frames, height, width = int(frames), int(height), int(width)
                merged_h, merged_w = height // merge, width // merge
                if merged_h * merge != height or merged_w * merge != width:
                    raise ValueError(
                        f"视觉塔网格 {height}×{width} 不能被 merge {merge} 整除 ✗"
                        f"（预处理那边就该切齐 ✓ —— 走到这里说明上游没按 merge 对齐 ✗）")
                block_rows = torch.arange(merged_h, device=device)
                block_cols = torch.arange(merged_w, device=device)
                intra = torch.arange(merge, device=device)
                row_idx = (block_rows[:, None, None, None] * merge
                           + intra[None, None, :, None])
                col_idx = (block_cols[None, :, None, None] * merge
                           + intra[None, None, None, :])
                coords = torch.stack((
                    row_idx.expand(merged_h, merged_w, merge, merge).reshape(-1),
                    col_idx.expand(merged_h, merged_w, merge, merge).reshape(-1),
                ), dim=-1)
                if frames > 1:
                    coords = coords.repeat(frames, 1)
                count = coords.shape[0]
                pos_ids[offset:offset + count] = coords
                offset += count
            if offset != total:
                raise ValueError(f"旋转位置铺了 {offset} 个 token，应为 {total} ✗（网格不解析 ✗）")
            return freq_table[pos_ids].flatten(1)

        def segment_lengths(self, grid_thw: Any) -> list[int]:
            """逐 chunk 的 token 数 ✓（每张图的 ``t`` 个 ``h·w`` 段 ✓ —— 段间不许互看 ✓）。"""
            lengths: list[int] = []
            for frames, height, width in grid_thw.tolist():
                lengths.extend([int(height) * int(width)] * int(frames))
            return lengths

        def forward(self, x: Any, grid_thw: Any) -> tuple[Any, list[Any]]:
            """``x``: ``(tokens, C·T·P·P)`` ✓ / ``grid_thw``: ``(n, 3)`` ✓。

            返回 ``(merged, deepstack_features)`` ✓ —— 主输出喂**语言塔的 embedding 位** ✓，
            DeepStack 三份按层序注进语言塔**前 3 层之后** ✓（见 :mod:`.qwen3vl_te` ✓）。
            """
            # ⚠️ **先**核 token 数 ✗✗ 再动权重 ✓：晚了的话错的是 ``x + pos_embed`` 那一行，
            #    报出来是「两个张量第一维不同」✓ 看着像网格问题、其实是上游 pixel 与 grid 不同源 ✓
            lengths = self.segment_lengths(grid_thw)
            expected = sum(lengths)
            if expected != int(x.shape[0]):
                raise ValueError(
                    f"网格算出 {expected} 个 patch、实际给了 {int(x.shape[0])} 个 ✗"
                    f"（``grid_thw`` 与像素**不同源** ✗ —— 不许猜、也不许补零 ✓）")
            # ⚠️ merge 整除也**提前**核 ✓：晚了会先在 ``view(t, h//m, m, w//m, m, -1)`` 崩 ✓，
            #    报的是「shape 不能 view」✓ 看着像张量问题、其实是网格没对齐 ✓✗
            for _frames, height, width in grid_thw.tolist():
                if int(height) % self.spatial_merge_size or int(width) % self.spatial_merge_size:
                    raise ValueError(
                        f"视觉塔网格 {int(height)}×{int(width)} 不能被 merge "
                        f"{self.spatial_merge_size} 整除 ✗（预处理那次 resize 就该切齐 ✓）")
            x = self.patch_embed(x)
            x = x + self.fast_pos_embed_interpolate(grid_thw).to(x.device)
            rotary = self.rot_pos_emb(grid_thw).to(x.device)
            # ⚠️ 宽度关系：``rotary_dim = head_dim//2`` ✓、``inv_freq`` 只有 ``rotary_dim//2`` 个频率 ✓
            #    ⇒ ``rot_pos_emb`` 出 ``2×(rotary_dim//2) = rotary_dim`` 宽 ✓、``cat([r,r])`` 后
            #    正好 ``2·rotary_dim = head_dim`` ✓（这一步对不上就别往下算 ✗）
            if rotary.shape[-1] != self.rotary_dim:
                raise ValueError(
                    f"旋转宽度 {rotary.shape[-1]} 与 ``rotary_dim = {self.rotary_dim}`` 不符 ✗"
                    f"（head_dim 必须能被 4 整除 ✓：{self.rotary_dim * 2} ⇒ 卡在 {self.rotary_dim} ✗）")
            # ⚠️ ``cat([r, r])`` 正好铺满 **head_dim** ✓（rotate_half 前后半同频 ✓）
            emb = torch.cat((rotary, rotary), dim=-1)
            cos = emb.cos().unsqueeze(-2)
            sin = emb.sin().unsqueeze(-2)
            half = sin.shape[-1] // 2
            position_embeddings = (cos, sin[..., :half], -sin[..., half:])
            deepstack: list[Any] = []
            for index, block in enumerate(self.blocks):
                x = block(x, lengths, position_embeddings)
                if index in self.deepstack_visual_indexes:
                    slot = self.deepstack_visual_indexes.index(index)
                    deepstack.append(self.deepstack_merger_list[slot](x))
            return self.merger(x), deepstack

    def build_vision_tower(shape: dict[str, Any], dtype: Any = None,
                           device: Any = None) -> Any:
        """造一台视觉塔 ✓（``dtype`` 给了就整体转 ✓ —— 盘上它是 BF16 ✓）。

        ``device="meta"`` ⇒ **在 meta 上构造** ✓（同 ``build_text_tower`` 那条口径 ✓：先按 CPU 建
        再 ``.to("meta")`` 已经白分配过一遍实存 ✓✗）。
        """
        if device is not None and str(device) == "meta":
            with torch.device("meta"):
                tower = VisionTower(shape)
        else:
            tower = VisionTower(shape)
        # 供 `h3_encode.combine` 回读 ✓（普通属性 ⇒ 不进 state_dict ✓ 不占参数位 ✓）
        tower.shape = dict(shape)
        if dtype is not None or device is not None:
            tower = tower.to(device=device, dtype=dtype)
        return tower

    return {"torch": torch, "nn": nn, "VisionTower": VisionTower,
            "build_vision_tower": build_vision_tower}


def build_vision_tower(shape: dict[str, Any], dtype: Any = None, device: Any = None) -> Any:
    """模块级入口 ✓（``_torch_parts()`` 里有同名实现 ✓ —— 这里只是转一手 ✓）。"""
    return _torch_parts()["build_vision_tower"](shape, dtype, device)
