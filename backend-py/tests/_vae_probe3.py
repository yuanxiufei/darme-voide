"""临时探针 3（用完即删 ✓）：按**真实使用比例**（块多 ✓ 重叠 25%）量分块近似度 ✓。"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from app.services.engine import vae_h3  # noqa: E402

PATH = r"D:\Comfy-Desktop\ComfyUI-Shared\models\vae\minimax_h3_video_vae_fp16.safetensors"
SIZE = 160


def build(**kw):
    cfg = vae_h3.H3VideoVAEConfig(weights=PATH, **kw)
    return vae_h3.build_h3_video_vae(cfg)


plain = build(tiling=False)
tiled = build(tiling=True, tile_size=64, tile_overlap_min=16)
print("plan:", tiled.split_tiles(SIZE))

g = torch.Generator().manual_seed(11)
x = torch.rand((1, 3, 5, SIZE, SIZE), generator=g) * 2 - 1
t0 = time.time()
za = plain.encode(x)
zb = tiled.encode(x)
print(f"encode {time.time() - t0:.1f}s rmse/std:", round(float((za - zb).pow(2).mean().sqrt()
                                                                / za.std()), 5))
# 解码分块：**同一份**潜变量 ✓ 只比解码 ✓
t0 = time.time()
ya = plain.decode(za)
yb = tiled.decode(za)
print(f"decode(同一 z) {time.time() - t0:.1f}s rmse/std:",
      round(float((ya - yb).pow(2).mean().sqrt() / ya.std()), 5),
      "max:", round(float((ya - yb).abs().max()), 4))
per_col = (ya - yb).abs().mean(dim=(0, 1, 2, 3))
print("dec err by column(step16):", [round(float(v), 4) for v in per_col[::16]])
print("dec err col 0/32/48/64/96/112/128:",
      [round(float(per_col[i]), 4) for i in (0, 32, 48, 64, 96, 112, 128)])
