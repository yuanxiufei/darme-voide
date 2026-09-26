"""临时探针 2（用完即删 ✓）：分块到底是"缝"还是"整体错位" ✓。"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from app.services.engine import vae_h3  # noqa: E402

PATH = r"D:\Comfy-Desktop\ComfyUI-Shared\models\vae\minimax_h3_video_vae_fp16.safetensors"
SIZE = 160


def build(**kw):
    return vae_h3.build_h3_video_vae(vae_h3.H3VideoVAEConfig(weights=PATH, **kw))


plain = build(tiling=False)
tiled = build(tiling=True, tile_size=128, tile_overlap_min=32)

# ── ① 分块**接线**正确性：把前向换成"网格对齐的池化" ✓ 这时分块必须**逐位**等价 ✓
probe = build(tiling=False)
probe._encode_moments = lambda x: F.avg_pool3d(  # type: ignore[method-assign]
    x, (4, 16, 16), stride=(4, 16, 16))
probe_t = build(tiling=True, tile_size=64, tile_overlap_min=16)
probe_t._encode_moments = lambda x: F.avg_pool3d(  # type: ignore[method-assign]
    x, (4, 16, 16), stride=(4, 16, 16))
g = torch.Generator().manual_seed(3)
xs = torch.rand((1, 3, 5, 96, 96), generator=g)
print("pool encode exact:", float((probe._encode_moments(xs) - probe_t.tiled_encode(xs)).abs().max()))

probe._decode_pixels = lambda z: F.interpolate(  # type: ignore[method-assign]
    z, scale_factor=(4, 16, 16), mode="nearest")
probe_t._decode_pixels = lambda z: F.interpolate(  # type: ignore[method-assign]
    z, scale_factor=(4, 16, 16), mode="nearest")
zs = torch.rand((1, 24, 2, 6, 6), generator=g)
print("pool decode exact:", float((probe._decode_pixels(zs) - probe_t.tiled_decode(zs)).abs().max()))

# ── ② 真权重下：误差是"缝"还是"整体错位" ✓
x = torch.rand((1, 3, 5, SIZE, SIZE), generator=g) * 2 - 1
t0 = time.time()
za = plain.encode(x)
zb = tiled.encode(x)
ya = plain.decode(za)
yb = tiled.decode(zb)
print(f"{time.time() - t0:.1f}s enc rmse/std:",
      float((za - zb).pow(2).mean().sqrt() / za.std()),
      "enc max:", float((za - zb).abs().max() / za.abs().max()))
print("dec rmse/std:", float((ya - yb).pow(2).mean().sqrt() / ya.std()),
      "dec max:", float((ya - yb).abs().max() / max(float(ya.abs().max()), 1e-6)))
per_col = (ya - yb).abs().mean(dim=(0, 1, 2, 3))
row = [round(float(v), 4) for v in per_col]
print("dec err by column (128 tile ⇒ 缝在 96~128):", row[::16])
print("dec err col 0 / 100 / 110 / max:", round(float(per_col[0]), 4),
      round(float(per_col[100]), 4), round(float(per_col[110]), 4), round(float(per_col.max()), 4))
per_col_z = (za - zb).abs().mean(dim=(0, 1, 2, 3))
print("enc err by column:", [round(float(v), 4) for v in per_col_z[::16]])
