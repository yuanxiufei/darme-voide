"""自检：**Flux VAE**（Z-Image / Flux 的 16 通道图片解码器 ✓ 2026-10-31）。

判据四类，都是**可判真假的事实** ✓：

* **口径类**：`latent_channels=16` ✓、`quant_conv=False` ✓、`scaling=0.3611` / `shift=0.1159` ✓；
  :func:`decode_latents` **必须先** ``x / scale + shift`` ✓（= `Flux.process_out` ✓）——
  用**恒等解码桩**把算术钉死 ✓（漏了不报错 ✓、写成 SDXL 那式 ``(x−shift)/scale`` ✗ 也不报错 ✓✗
  ⇒ 只能靠这条自检抓 ✓）；
* **键名类**：:func:`flux_vae_key_names` 给 **244** 键 ✓，且 ``quant_conv`` / ``post_quant_conv``
  一个都不许有 ✓（244 = SDXL 的 248 − 4 ✓ —— 这是**唯一**结构差异 ✓）；
* **真权重类**：本机 ``vae/ae.safetensors`` 的头**逐键比对** ✓（名字 + 形状全等才绿 ✓，只读头不载张量 ✗）；
* **预检类**：:func:`shape_problems` 对真形状必须空 ✓；把 ``decoder.conv_in`` 的潜通道换成 **4**
  ⇒ 必须报「拿成了 SDXL VAE」✓（不报就说明这条预检形同虚设 ✗）。

⚠️ 没装 torch / 扫不到真权重 ⇒ 显式 SKIP ✓（**没跑 ≠ 绿** ✗）。
⚠️ ``ComfyUI`` 是 GPL-3.0 ⇒ 一行都不抄 ✗（只当规格书读 ✓）。

运行::

    ./.venv/Scripts/python.exe tests/engine_vae_flux_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import sdxl_vae as sdxl_mod  # noqa: E402
from app.services.engine import vae_flux as mod  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def case_facts() -> None:
    """① 口径类 ✓（这些数是核过的 ✓ 改一个就该红 ✓）。"""
    facts = mod.FLUX_VAE_FACTS
    check("① 潜通道 16 ✓（Z-Image/Flux ✓；4 是 SDXL ✗）",
          facts["latentChannels"] == 16, facts["latentChannels"])
    check("①′ 缩放 0.3611 / 平移 0.1159 ✓（`latent_formats.Flux` ✓）",
          facts["scalingFactor"] == 0.3611 and facts["shiftFactor"] == 0.1159, facts)
    check("①″ 真权重 244 键 ✓", facts["keyCount"] == 244, facts["keyCount"])
    spec = mod.flux_vae_config()
    check("①‴ 配置里 `quant_conv=False` ✓（Flux 没有那 4 个 1×1 卷积 ✓）",
          spec.quant_conv is False, spec.quant_conv)
    check("①⁗ 潜空间下采样 8 ✓（与 DiT 的 `vae_scale` 同一口径 ✓）",
          facts["spatialScale"] == 8, facts["spatialScale"])


def case_key_names() -> None:
    """② 键名类 ✓（纯结构推 ✓ ⇒ 不靠真权重也能验 ✓）。"""
    keys = mod.flux_vae_key_names()
    check(f"② 恰 244 键 ✓（实测 {len(keys)} ✓）", len(keys) == 244, len(keys))
    bad = [key for key in keys if "quant_conv" in key]
    check("②′ ⭐ **一个 `quant_conv` / `post_quant_conv` 都没有** ✓✗（有就说明错成了 SDXL ✓）",
          not bad, bad)
    sdxl_keys = sdxl_mod.sdxl_vae_key_names(sdxl_mod.SdxlVaeConfig())
    check("②″ 与 SDXL 的差正好是 **4 个** quant_conv 键 ✓（不是别的差异 ✓）",
          set(sdxl_keys) - set(keys) == {
              "quant_conv.weight", "quant_conv.bias",
              "post_quant_conv.weight", "post_quant_conv.bias"},
          sorted(set(sdxl_keys) - set(keys)))


def case_scale_arithmetic() -> None:
    """③ 反缩放 ✓ —— 用**恒等解码桩**把算术钉死（漏了不报错，只能这么验 ✓）。"""
    if not sdxl_mod.has_torch():
        skip("没装 torch ⇒ 反缩放算术不跑 ✓（**没跑 ≠ 绿** ✗）")
        return
    import torch  # noqa: PLC0415

    class _Identity:
        """`decode()` 原样返回输入 ⇒ 输出就该是反缩放后的潜变量 ✓。"""

        config = mod.flux_vae_config()

        def decode(self, latents: Any) -> Any:
            return latents

    latents = torch.full((1, 16, 2, 2), 0.5, dtype=torch.float32)
    out = mod.decode_latents(_Identity(), latents, to_unit=False)
    want = 0.5 / 0.3611 + 0.1159
    check("③ ⭐⭐ 解码前先 `x / 0.3611 + 0.1159` ✓（= `Flux.process_out` ✓；漏了不报错 ✓ 只是偏色 ✓）",
          abs(float(out[0, 0, 0, 0]) - want) < 1e-5, (float(out[0, 0, 0, 0]), want))
    #: ⚠️ 反向：把式子写成**减 shift 再除**（SDXL 的等价式 ✗）必须差得抓得住 ✓✗ ——
    #: 两条在 SDXL 上（shift=0）**完全等价** ✓，只有到这里才分得开 ✓。
    wrong = (0.5 - 0.1159) / 0.3611
    check("③‴ 反向：`(x − shift) / scale` ✗ 与真值差 ≈0.437 ✓（这个差在 SDXL 上恒为 0 ⇒ 试不出来 ✓✗）",
          abs(want - wrong) > 0.4, (want, wrong, abs(want - wrong)))
    check("③′ 桩没被偷偷改形状 ✓（反缩放是逐元素 ✓）", tuple(out.shape) == (1, 16, 2, 2),
          tuple(out.shape))


def case_shape_problems() -> None:
    """④ 预检类 ✓：真形状必须空；换成 4 通道必须报「拿成了 SDXL」✓。"""
    real = {key: tuple(shape) for key, shape in _real_shapes().items()} if _real_shapes() else {}
    if not real:
        skip("没扫到 `vae/ae.safetensors` ⇒ 预检类只做**人造形状**那一半 ✓")
        fake = {key: (1,) for key in mod.flux_vae_key_names()}
        fake["decoder.conv_in.weight"] = (512, 16, 3, 3)
        fake["encoder.conv_out.weight"] = (32, 512, 3, 3)
        check("④ 人造形状（16 通道 ✓）⇒ 预检**空** ✓", not mod.shape_problems(fake),
              mod.shape_problems(fake))
        fake["decoder.conv_in.weight"] = (512, 4, 3, 3)
        problems = mod.shape_problems(fake)
        check("④′ 潜通道换成 4 ⇒ 报「拿成了 SDXL VAE」✓✗",
              any("SDXL" in text for text in problems), problems)
        return

    check("④ 真权重形状 ⇒ 预检**空** ✓（有字就说明配置与真权重不符 ✗）",
          not mod.shape_problems(real), mod.shape_problems(real))
    broken = dict(real)
    broken["decoder.conv_in.weight"] = (512, 4, 3, 3)
    problems = mod.shape_problems(broken)
    check("④′ 潜通道换成 4 ⇒ 报「拿成了 SDXL VAE」✓✗",
          any("SDXL" in text for text in problems), problems)


def case_real_weights() -> None:
    """⑤ 真权重逐键 ✓（名字 + 形状全等 ✓；meta 设备建壳 ⇒ 不占内存 ✓）。"""
    if not sdxl_mod.has_torch():
        skip("没装 torch ⇒ 真权重对齐不跑 ✓（**没跑 ≠ 绿** ✗）")
        return
    shapes = _real_shapes()
    if not shapes:
        skip("没扫到 `vae/ae.safetensors` ⇒ 真权重对齐不跑 ✓（**没跑 ≠ 绿** ✗）")
        return

    import torch  # noqa: PLC0415

    with torch.device("meta"):
        model = mod.build_flux_vae()
    mine = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    missing, extra = sorted(set(shapes) - set(mine)), sorted(set(mine) - set(shapes))
    check("⑤ ⭐⭐ 键集与真权重**全等** ✓✗（块数 / 键序错一处必红 ✓）",
          not missing and not extra, {"缺": missing[:5], "多": extra[:5]})
    mismatched = sorted(key for key in set(shapes) & set(mine) if tuple(shapes[key]) != mine[key])
    check("⑤′ ⭐⭐ 逐键形状全等 ✓✗（通道账错一层就在这条里 ✓）", not mismatched,
          [(key, shapes[key], mine[key]) for key in mismatched[:5]])
    total = sum(torch.Size(tuple(shape)).numel() for shape in shapes.values())
    check("⑤″ 参数量与真权重**算出来的**一致 ✓",
          sum(torch.Size(shape).numel() for shape in mine.values()) == total, total)


def _real_shapes() -> dict[str, tuple[int, ...]]:
    """扫本机 ``vae/ae.safetensors`` 的键 → 形状 ✓（**只读头** ✗ 不载张量 ✓；扫不到给空 ✓）。"""
    path, _why = _find_ae()
    if path is None:
        return {}
    from app.services.engine.safetensors import read_header  # noqa: PLC0415

    _, header = read_header(path)
    return {key: tuple(int(v) for v in header[key]["shape"]) for key in header
            if key != "__metadata__"}


def _find_ae() -> tuple[Path | None, str]:
    """找本机 Flux VAE —— **走本仓扫描器** ✓，不写死机器路径 ✗（同 ``engine_sdxl_vae_test`` ✓）。"""
    try:
        from app.services import local_model_scan as scan  # noqa: PLC0415
    except Exception as error:  # noqa: BLE001
        return None, f"导不进本仓扫描器（{error}）⇒ 真权重对齐不跑 ✓"
    roots = list(scan.get_default_roots())
    # ⚠️ `kinds` 只有 video/image/text/audio ✓（`local_model_scan._SCAN_KINDS` ✓）——
    #    没有 "vae" 这一档 ✗ ⇒ **按文件名在 image 档里筛** ✓（`ae.safetensors` 归 image ✓）。
    try:
        scanned = scan.scan_local_models({"roots": roots, "kinds": ["image"], "maxFiles": 40000})
    except Exception as error:  # noqa: BLE001
        return None, f"扫描器报错（{error}）⇒ 真权重对齐不跑 ✓"
    hits = sorted((item for item in scanned["models"]
                   if Path(str(item.get("filename") or "")).name.lower() == "ae.safetensors"),
                  key=lambda item: str(item.get("path") or ""))
    if not hits:
        cut = ("（⚠️ 扫描被 `maxFiles` 截断了 ⇒「扫不到」也可能只是**没扫完** ✓ 不许当通过 ✗）"
               if scanned.get("truncated") else "")
        return None, (f"扫描器扫了 {len(roots)} 个根 / {scanned.get('total')} 个图类权重，"
                      f"没有 `ae.safetensors` ⇒ 真权重对齐不跑 ✓{cut}")
    return Path(str(hits[0]["path"])), f"扫描器命中 {len(hits)} 份"


def main() -> int:
    case_facts()
    case_key_names()
    case_scale_arithmetic()
    case_shape_problems()
    case_real_weights()

    failed = [row for row in _RESULTS if not row[1]]
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            print(f"      实测: {detail}")
    for reason in _SKIPS:
        print(f"SKIP  {reason}")
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
