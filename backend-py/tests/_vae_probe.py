"""临时探针（用完即删 ✓）：真权重装载 + 数值自检。"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from app.services.engine import vae_h3  # noqa: E402

PATH = r"D:\Comfy-Desktop\ComfyUI-Shared\models\vae\minimax_h3_video_vae_fp16.safetensors"

t0 = time.time()
vae = vae_h3.build_h3_video_vae(vae_h3.H3VideoVAEConfig(weights=PATH, tiling=False))
print(f"load {time.time() - t0:.1f}s complete={vae.loadReport.complete} "
      f"missing={len(vae.loadReport.missing)} extra={len(vae.loadReport.unexpected)} "
      f"shapes={len(vae.loadReport.shapeMismatch)} dtype={next(vae.parameters()).dtype}")
print("latents_mean[:4]", [round(float(v), 5) for v in vae.latents_mean[:4]])

g = torch.Generator().manual_seed(7)
x = torch.rand((1, 3, 5, 32, 32), generator=g) * 2 - 1

t0 = time.time()
z = vae.encode(x)
print(f"encode {time.time() - t0:.1f}s shape={tuple(z.shape)} "
      f"expect={vae_h3.latents_for_frames(5)}")
print("z mean/std", round(float(z.mean()), 4), round(float(z.std()), 4))
print("verify_latents_stats ok:", vae_h3.verify_latents_stats(z)["ok"],
      vae_h3.verify_latents_stats(z)["maxAbsMeanDev"], vae_h3.verify_latents_stats(z)["maxRelStdDev"])

t0 = time.time()
y = vae.decode(z)
print(f"decode {time.time() - t0:.1f}s shape={tuple(y.shape)} "
      f"expect={vae.decode_output_shape(tuple(z.shape))}")
print("y range", round(float(y.min()), 4), round(float(y.max()), 4),
      "finite", bool(torch.isfinite(y).all()))

# 因果性：17 帧那一段的潜变量，必须与 34 帧跑出来的**前两格**逐位一致
x17 = x[:, :, :5].repeat(1, 1, 4, 1, 1)[:, :, :17]
x34 = torch.cat([x17, torch.rand((1, 3, 17, 32, 32), generator=g) * 2 - 1], dim=2)
z17 = vae.encode(x17)
z34 = vae.encode(x34)
print("causal 17/34:", tuple(z17.shape), tuple(z34.shape),
      "maxdiff", float((z17 - z34[:, :, :z17.shape[2]]).abs().max()))

# 分块 vs 不分块（**近似** ✓ 不是等价 ✓）
xa = torch.rand((1, 3, 5, 96, 96), generator=g) * 2 - 1
tiled = vae_h3.build_h3_video_vae(vae_h3.H3VideoVAEConfig(
    weights=PATH, tiling=True, tile_size=64, tile_overlap_min=16))
za = vae.encode(xa)
zb = tiled.encode(xa)
print("tiling encode:", tuple(za.shape), tuple(zb.shape),
      "rel", float((za - zb).abs().max() / za.abs().max()))
ya = vae.decode(za)
yb = tiled.decode(zb)
print("tiling decode:", tuple(ya.shape), tuple(yb.shape),
      "rel", float((ya - yb).abs().max() / max(float(ya.abs().max()), 1e-6)))
