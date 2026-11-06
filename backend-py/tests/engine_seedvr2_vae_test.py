"""SEEDVR2 VAE（``vae_seedvr2``，ByteDance ``VideoAutoencoderKL``）的**结构与数值**自检 ✓。

判据八层（前七层不依赖盘上权重，CI 无模型也照跑 ✓）：

① **形状反推**：从 250 个真键反推 ``block_out_channels=(128,256,512,512)`` / 每级 2 个
   ResNet / 时间下采样 2 次 / 潜变量 16 通道 / 空间 8× / 时间 4× ✓；且 ``conv_in`` 出通道
   必须**等于**第 0 级通道（✗ 把它当成「第 0 级」再 append 会凭空多出一级 ⇒ 空间比 8 变 16 ✓）。
② **反推守门**：不是这份结构就**当场人话拒绝** ✓（少 ``conv_in`` / 4 维核 / 全部时间核 t=1
   的 2D 图 VAE / 编解码时间口径不对称 ✓ 四种坏法各一条判据）。
③ **结构自洽**：通道不被组数整除 / 时间下采样 0 次 / 级数 <2 都判问题 ✓。
④ **键对账**：``build(infer_shape(...))`` 后流式装载必须 ``complete`` ✓（0 缺 0 错形 0 多余 ✓）。
⑤ **帧数铁律**：``T = 4n+1`` ✓（``(T-1)%4`` 非 0 当场拒且消息里给最近的两个合法值 ✓），
   ``latent_frames`` 与 ``pixel_frames`` 严格互逆 ✓。
⑥ **小形状自洽**：3 级小网（无真权重 ✓）能编码/解码、形状与时序口径自洽 ✓。
⑦ **因果性**：扰动**末**帧不许改**首**潜帧 ✓（编码与解码两侧都查 ✓）—— 这条抓「时间 padding
   写成零填充 / 注意力跨帧」这类**形状全对、语义全错**的实现 ✓。
⑧ **黄金数值**：与 ComfyUI ``comfy/ldm/seedvr/vae.py`` 逐位对拍得到的基准（fp32/CPU、
   输入 ``manual_seed(1234)`` 的 ``rand(1,3,5,64,48)*2-1``，**encode/decode maxdiff 均 0.0** ✓）
   写死成常数 ✓ —— 抓「结构照抄、口径差一点」的数值漂移 ✓。

另有两条**反向验证**：把三处口径各改坏一次（时间 padding 换零填 / ``remove_head`` 恒等 /
组归一化跨帧），断言⑦⑧ 的判据**真的会红** ✓ —— 证明判据有分辨力，不是「两份都错也一样绿」✓。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: ⚠️ cp936 控制台下第一个 ``✓`` 就会 ``UnicodeEncodeError`` ⇒ 判据还没跑就崩 ✓✗。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from app.services.engine import latent_formats as lf  # noqa: E402
from app.services.engine import vae_seedvr2 as sv  # noqa: E402
from app.services.engine.weights import load_module_weights  # noqa: E402

FAILS: list[str] = []
CHECKS: list[str] = []
SKIPS: list[str] = []

REAL = Path("D:/Comfy-Desktop/ComfyUI-Shared/models/vae/SEEDVR2/ema_vae_fp16.safetensors")

#: 3 级小网 ✓（空间比 4 ✓、时间比 2 ✓、潜变量 4 通道 ✓）—— 全部自己造，不碰盘 ✓。
SMALL = sv.SeedVR2VaeShape(
    block_out_channels=(8, 16, 16), layers_per_block=1, norm_num_groups=8,
    latent_channels=4, temporal_scale_num=1,
)

#: ⑦ 基准（来源见模块头 ⑧ ✓）：``z``/``y`` 的四个统计量 ＋ 前 8 个元素 ✓。
GOLDEN_Z = (0.274383, 0.867941, -2.503462, 3.169164)
GOLDEN_Y = (0.034074, 0.323139, -1.270896, 1.415766)
GOLDEN_Z8 = (0.195269, 2.053747, -0.11505, 1.729634, 1.095941, 0.119279, 0.77469, 1.832254)
GOLDEN_Y8 = (-0.064767, -0.028936, 0.227028, -0.26544, -0.229368, -0.233318,
             -0.353012, -0.35934)


def check(name: str, ok: object, detail: object = None) -> None:
    CHECKS.append(name)
    print(f"  {'✓' if ok else '✗'} {name}")
    if not ok:
        FAILS.append(name)
        if detail is not None:
            print(f"      {detail}")


def skip(name: str, why: str) -> None:
    SKIPS.append(name)
    print(f"  · {name} —— 跳过：{why}")


def _real_shapes() -> dict[str, tuple[int, ...]]:
    from app.services.engine import safetensors as st

    return {k: tuple(e.shape) for k, e in st.inspect(REAL).tensors.items()}


def _real_vae() -> sv.SeedVR2Vae:
    #: ✗ 盘上是 fp16 ✓；基准是在 **fp32/CPU** 上取的 ✓ ⇒ 这里必须显式 ``dtype=float32`` ✓，
    #: 否则权重按 fp16 落进模型 ✓，数值差 1e-2 量级、看着像"实现算错了" ✓。
    vae = sv.build(sv.infer_shape(_real_shapes()), dtype=torch.float32)
    report = load_module_weights(vae, REAL, dtype=torch.float32)
    assert report.complete, report.to_dict()
    return vae


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).abs().max())


def _err(call) -> str:
    try:
        call()
    except Exception as exc:  # noqa: BLE001 —— 判据要看的就是"报没报、报得对不对" ✓
        return str(exc)
    return ""


# ------------------------------------------------------------------ ① 反推 ----
def case_infer() -> None:
    print("① 从真键反推结构")
    if not REAL.exists():
        skip("① 形状反推", f"权重未就位：{REAL}")
        return
    shape = sv.infer_shape(_real_shapes())
    check("① 级数与通道 (128,256,512,512)",
          shape.block_out_channels == (128, 256, 512, 512), shape.block_out_channels)
    check("① 每级 2 个 ResNet", shape.layers_per_block == 2, shape.layers_per_block)
    check("① 时间下采样 2 次 ⇒ ÷4", shape.temporal_downscale_ratio == 4,
          shape.temporal_scale_num)
    check("① 空间下采样 ÷8", shape.spatial_downscale_ratio == 8)
    check("① 潜变量 16 通道", shape.latent_channels == 16, shape.latent_channels)
    check("① 缩放 0.9152 与族表一致",
          abs(shape.scale_factor - lf.get_latent_format("seedvr2").scale_factor) < 1e-9,
          (shape.scale_factor, lf.get_latent_format("seedvr2").scale_factor))
    check("① 反推留了凭据", len(shape.evidence) >= 6, shape.evidence)


# ------------------------------------------------------------- ② 反推守门 ----
def case_infer_guard() -> None:
    print("② 反推守门（坏权重必须人话拒绝）")
    if not REAL.exists():
        skip("② 反推守门", f"权重未就位：{REAL}")
        return
    shapes = _real_shapes()

    missing = {k: v for k, v in shapes.items() if not k.startswith("encoder.conv_in")}
    msg = _err(lambda: sv.infer_shape(missing))
    check("② 少 conv_in ⇒ 明说不是这份权重", "SEEDVR2" in msg, msg)

    flat = dict(shapes)
    flat["encoder.conv_in.weight"] = tuple(flat["encoder.conv_in.weight"][-4:])
    msg = _err(lambda: sv.infer_shape(flat))
    check("② 4 维核 ⇒ 明说不是 3D 视频 VAE", "3D" in msg, msg)

    image = {}
    for key, value in shapes.items():
        if key.endswith("downsamplers.0.conv.weight") and len(value) == 5:
            image[key] = (value[0], value[1], 1, value[3], value[4])
        else:
            image[key] = value
    msg = _err(lambda: sv.infer_shape(image))
    check("② 全是纯空间核 ⇒ 认成 2D 图 VAE 并拒绝", "2D" in msg or "①图" in msg, msg)

    asym = dict(shapes)
    asym["decoder.up_blocks.0.upsamplers.0.upscale_conv.weight"] = (2048, 512, 1, 1, 1)
    msg = _err(lambda: sv.infer_shape(asym))
    check("② 编解码时间口径不对称 ⇒ 当场拒", "不对称" in msg, msg)


# ------------------------------------------------------------- ③ 结构自洽 ----
def case_shape_problems() -> None:
    print("③ 结构自洽检查")
    bad_groups = sv.SeedVR2VaeShape(block_out_channels=(8, 12, 12), layers_per_block=1,
                                    norm_num_groups=8, latent_channels=4,
                                    temporal_scale_num=1)
    check("③ 通道不被组数整除 ⇒ 报问题",
          any("整除" in p for p in sv.shape_problems(bad_groups)),
          sv.shape_problems(bad_groups))

    no_temporal = sv.SeedVR2VaeShape(block_out_channels=(8, 16), layers_per_block=1,
                                     norm_num_groups=8, latent_channels=4,
                                     temporal_scale_num=0)
    check("③ 时间下采样 0 次 ⇒ 报问题",
          any("4n+1" in p for p in sv.shape_problems(no_temporal)),
          sv.shape_problems(no_temporal))

    check("③ 小网本身自洽", sv.shape_problems(SMALL) == (), sv.shape_problems(SMALL))
    check("③ 不自洽 ⇒ build 当场拒", "整除" in _err(lambda: sv.build(bad_groups)))


# --------------------------------------------------------------- ④ 键对账 ----
def case_keys() -> None:
    print("④ 真权重键对账")
    if not REAL.exists():
        skip("④ 键对账", f"权重未就位：{REAL}")
        return
    dense = sv.SeedVR2Vae(sv.infer_shape(_real_shapes()))
    report = load_module_weights(dense, REAL)
    check("④ 装载 complete（0 缺 0 错形）", report.complete, report.to_dict())
    check("④ 没有多余键", not report.unexpected, report.unexpected[:6])
    check("④ 键数 250", report.sourceKeys == 250, report.sourceKeys)
    total = sum(p.numel() for p in dense.parameters())
    check("④ 参数量 250 648 227", total == 250_648_227, total)


# ------------------------------------------------------------- ⑤ 帧数铁律 ----
def case_frames() -> None:
    print("⑤ 帧数铁律 T = 4n+1")
    check("⑤ 5 帧 ⇒ 2 个潜帧", sv.latent_frames(5) == 2, sv.latent_frames(5))
    check("⑤ 1 帧 ⇒ 1 个潜帧", sv.latent_frames(1) == 1, sv.latent_frames(1))
    check("⑤ 9 帧 ⇒ 3 个潜帧", sv.latent_frames(9) == 3, sv.latent_frames(9))
    msg = _err(lambda: sv.latent_frames(4))
    check("⑤ 4 帧 ⇒ 当场拒并给出最近合法值", "4n+1" in msg and "5" in msg, msg)
    check("⑤ 潜帧→像素帧互逆",
          all(sv.pixel_frames(sv.latent_frames(n)) == n for n in (1, 5, 9, 33)))
    check("⑤ 小网口径 ÷2", sv.latent_frames(5, SMALL) == 3, sv.latent_frames(5, SMALL))
    noclass = sv.pixel_frames(3, SMALL)
    check("⑤ 小网潜 3 ⇒ 像素 5", noclass == 5, noclass)

    vae = sv.SeedVR2Vae(SMALL)
    check("⑤ 宽高不整除 ⇒ 当场拒", "整数倍" in _err(lambda: vae.latent_size(10, 12)))
    check("⑤ 宽高整除 ⇒ 通过", vae.latent_size(16, 8) == (4, 2), vae.latent_size(16, 8))


# ----------------------------------------------------------- ⑥ 小形状自洽 ----
def case_small() -> None:
    print("⑥ 小形状自洽（不读盘）")
    torch.manual_seed(7)
    vae = sv.build(SMALL)
    x = torch.rand(2, 3, 5, 16, 8) * 2 - 1
    with torch.no_grad():
        z = vae.encode(x)
        y = vae.decode(z)
    check("⑥ 编码形状 (2,4,3,4,2)", tuple(z.shape) == (2, 4, 3, 4, 2), tuple(z.shape))
    check("⑥ 解码形状回到 (2,3,5,16,8)", tuple(y.shape) == (2, 3, 5, 16, 8), tuple(y.shape))
    check("⑥ 全 finite", bool(torch.isfinite(z).all() and torch.isfinite(y).all()))
    check("⑥ 5 帧进 3 潜帧出", z.shape[2] == 3)

    msg = _err(lambda: vae.encode(torch.rand(1, 3, 4, 16, 8)))
    check("⑥ 4 帧输入 ⇒ 编码前就拒", "4n+1" in msg, msg)
    msg = _err(lambda: vae.encode(torch.rand(1, 4, 5, 16, 8)))
    check("⑥ 通道不对 ⇒ 明说应为 3", "通道应为 3" in msg, msg)
    msg = _err(lambda: vae.decode(torch.rand(1, 5, 3, 4, 2)))
    check("⑥ 解码通道不对 ⇒ 明说应为 4", "通道应为 4" in msg, msg)


# --------------------------------------------------------------- ⑦ 因果性 ----
def case_causal() -> None:
    print("⑦ 因果性（未来帧不许影响过去）")
    torch.manual_seed(11)
    vae = sv.build(SMALL)
    x = torch.rand(1, 3, 5, 16, 8) * 2 - 1
    bumped = x.clone()
    bumped[:, :, -1] += 0.75
    with torch.no_grad():
        za, zb = vae.encode(x), vae.encode(bumped)
    check("⑦ 编码：首潜帧不受末帧影响", _maxdiff(za[:, :, 0], zb[:, :, 0]) < 1e-6,
          _maxdiff(za[:, :, 0], zb[:, :, 0]))
    check("⑦ 编码：末潜帧确实变了", _maxdiff(za[:, :, -1], zb[:, :, -1]) > 1e-3,
          _maxdiff(za[:, :, -1], zb[:, :, -1]))

    lat = torch.rand(1, 4, 3, 4, 2)
    jumped = lat.clone()
    jumped[:, :, -1] += 1.0
    with torch.no_grad():
        ya, yb = vae.decode(lat), vae.decode(jumped)
    check("⑦ 解码：首像素帧不受末潜帧影响", _maxdiff(ya[:, :, 0], yb[:, :, 0]) < 1e-6,
          _maxdiff(ya[:, :, 0], yb[:, :, 0]))
    check("⑦ 解码：末像素帧确实变了", _maxdiff(ya[:, :, -1], yb[:, :, -1]) > 1e-3,
          _maxdiff(ya[:, :, -1], yb[:, :, -1]))


# ----------------------------------------------------------- ⑧ 黄金数值 ----
def case_golden() -> None:
    print("⑧ 黄金数值（与 ComfyUI 参考实现逐位对拍）")
    if not REAL.exists():
        skip("⑧ 黄金数值", f"权重未就位：{REAL}")
        return
    vae = _real_vae()
    torch.manual_seed(1234)
    x = torch.rand(1, 3, 5, 64, 48) * 2 - 1
    with torch.no_grad():
        z = vae.encode(x)
        y = vae.decode(z)
    zs = (float(z.mean()), float(z.std()), float(z.min()), float(z.max()))
    ys = (float(y.mean()), float(y.std()), float(y.min()), float(y.max()))
    check("⑧ 潜变量统计量", max(abs(a - b) for a, b in zip(zs, GOLDEN_Z)) < 2e-3,
          (tuple(round(v, 6) for v in zs), GOLDEN_Z))
    check("⑧ 像素统计量", max(abs(a - b) for a, b in zip(ys, GOLDEN_Y)) < 2e-3,
          (tuple(round(v, 6) for v in ys), GOLDEN_Y))
    check("⑧ 潜变量前 8 个值",
          max(abs(float(a) - b) for a, b in zip(z.reshape(-1)[:8], GOLDEN_Z8)) < 2e-3,
          tuple(round(float(v), 6) for v in z.reshape(-1)[:8]))
    check("⑧ 像素前 8 个值",
          max(abs(float(a) - b) for a, b in zip(y.reshape(-1)[:8], GOLDEN_Y8)) < 2e-3,
          tuple(round(float(v), 6) for v in y.reshape(-1)[:8]))

    model_space = sv.encode_latents(vae, x)
    check("⑧ encode_latents 走 ×0.9152",
          _maxdiff(model_space, z * 0.9152) < 1e-6, _maxdiff(model_space, z))
    back = sv.decode_latents(vae, model_space)
    #: ✗ ``×0.9152`` 与 ``÷0.9152`` 在 fp32 里不是精确互逆 ✓（相对误差 ~1e-7 ✓），
    #: 解码器会把它放大到 1e-5 量级 ✓ ⇒ 这里给 1e-4 的门槛 ✓，不是"实现不一致" ✗。
    check("⑧ decode_latents 与裸解码一致（换算自洽）",
          _maxdiff(back, y) < 1e-4, _maxdiff(back, y))
    check("⑧ 缩放方向写反会被抓出来",
          _maxdiff(sv.decode_latents(vae, z / 0.9152), y) > 1e-3,
          _maxdiff(sv.decode_latents(vae, z / 0.9152), y))


# --------------------------------------------------------- ⑨ 反向验证 ----
def case_reverse() -> None:
    """三处口径各改坏一次 ⇒ ⑦⑧ 的判据必须变红 ✓。"""
    print("⑨ 反向验证（改坏口径 ⇒ 判据必须红）")
    torch.manual_seed(11)
    vae = sv.build(SMALL)
    x = torch.rand(1, 3, 5, 16, 8) * 2 - 1

    original_extend = sv._extend_head
    original_remove = sv._remove_head
    original_norm = sv._GroupNorm5d.forward
    try:
        sv._extend_head = lambda t, times: F.pad(t, (0, 0, 0, 0, times, 0))
        with torch.no_grad():
            zero_pad = vae.encode(x)
        sv._extend_head = original_extend
        with torch.no_grad():
            causal = vae.encode(x)
        check("⑨ 时间 padding 写成零填 ⇒ 数值变（不是等价改写）",
              _maxdiff(zero_pad, causal) > 1e-4, _maxdiff(zero_pad, causal))

        sv._remove_head = lambda t, times=1: t
        with torch.no_grad():
            keep_all = vae.decode(torch.rand(1, 4, 3, 4, 2))
        sv._remove_head = original_remove
        with torch.no_grad():
            dropped = vae.decode(torch.rand(1, 4, 3, 4, 2))
        check("⑨ remove_head 恒等 ⇒ 帧数变多（4n+1 失效）",
              keep_all.shape[2] > dropped.shape[2],
              (tuple(keep_all.shape), tuple(dropped.shape)))

        sv._GroupNorm5d.forward = lambda self, t: F.group_norm(
            t.to(self.weight.dtype), self.num_groups, self.weight, self.bias, self.eps)
        with torch.no_grad():
            cross_frame = vae.encode(x)
        sv._GroupNorm5d.forward = original_norm
        check("⑨ 组归一化跨帧 ⇒ 数值变（逐帧统计才是对的）",
              _maxdiff(cross_frame, causal) > 1e-5, _maxdiff(cross_frame, causal))
    finally:
        sv._extend_head = original_extend
        sv._remove_head = original_remove
        sv._GroupNorm5d.forward = original_norm


# ------------------------------------------------------------- ⑩ 人话回执 ----
def case_readout() -> None:
    print("⑩ 回执人话")
    text = sv.readout(SMALL)
    check("⑩ 写出空间/时间比", "÷4" in text and "÷2" in text, text)
    check("⑩ 写出 4n+1", "4n+1" in text, text)
    check("⑩ 写明逐帧注意力", "逐帧" in text, text)


# ---------------------------------------------------- ⑪ 接线（torch_backend） ----
def case_backend() -> None:
    """VAE 挂进 ``TorchBackend`` 之后的**两个方向** ✓（像素进 / 像素出 ✓）。"""
    print("⑪ 接线：attach_seedvr2_vae ＋ encode_pixels ＋ decode")
    from app.services.engine import torch_backend as tb

    backend = tb.TorchBackend(device="cpu")
    #: ⚠️ 没挂就调 ⇒ 必须**点名 seedvr2** ✓（不能只说"没挂 VAE" ✗：那样接线的人会去挂 Wan ✗）。
    msg = _err(lambda: backend.encode_pixels(torch.rand(1, 3, 5, 16, 16)))
    check("⑪ 未挂 ⇒ 拒绝并点名 seedvr2", "seedvr2" in msg and "attach_seedvr2_vae" in msg, msg)

    if not REAL.exists():
        skip("⑪ 接线", f"权重未就位：{REAL}")
        return
    detail = backend.attach_seedvr2_vae(str(REAL), dtype=torch.float32)
    check("⑪ 装上 250 键 complete", detail["complete"] and detail["loadedKeys"] == 250,
          (detail["complete"], detail["loadedKeys"]))
    check("⑪ kind 说真话", detail["kind"] == "seedvr2-real", detail["kind"])
    check("⑪ 回执带换算与帧号口径",
          "0.9152" in detail["note"] and "4n+1" in detail["note"], detail["note"])

    torch.manual_seed(99)
    pixels = torch.rand(1, 3, 5, 32, 24) * 2 - 1
    encoded = backend.encode_pixels(pixels)
    check("⑪ encode_pixels 出 (1,16,2,4,3)",
          encoded["shape"] == [1, 16, 2, 4, 3], encoded["shape"])
    check("⑪ 回执说清「已在模型空间」", encoded["modelSpace"] is True, encoded)
    check("⑪ 回执带空间/时间比与帧号",
          (encoded["spatialScale"], encoded["temporalScale"], encoded["latentFrames"])
          == (8, 4, 2),
          (encoded["spatialScale"], encoded["temporalScale"], encoded["latentFrames"]))
    check("⑪ 潜变量与裸编码 ×0.9152 一致",
          _maxdiff(encoded["latents"], backend._vae.encode(pixels) * 0.9152) < 1e-6,
          _maxdiff(encoded["latents"], backend._vae.encode(pixels)))

    #: `zero_to_one` 只是替调用方做 `×2−1` ✓ —— 同一批像素两种口径必须**逐值相同** ✓
    #: （写成 `×2+1` / 忘记减 ⇒ 潜变量整体偏移、画面发灰、不报错 ✗）。
    same = backend.encode_pixels(pixels.add(1).div(2), pixel_range="zero_to_one")
    check("⑪ [0,1] 口径自动归一", _maxdiff(same["latents"], encoded["latents"]) < 1e-6,
          _maxdiff(same["latents"], encoded["latents"]))
    msg = _err(lambda: backend.encode_pixels(pixels, pixel_range="0-255"))
    check("⑪ 口径写错 ⇒ 明说只认两档", "minus_one_to_one" in msg, msg)

    #: 往返：像素 → 潜 → 像素 ✓（解码走 `÷0.9152` ✓）。VAE 未经修复训练 ⇒ 只核**形状/口径** ✓，
    #: 不核像素相似度 ✗（那不是这条判据要管的事 ✓）。
    decoded = backend.decode(encoded["latents"], type("P", (), {"frames": 5})(), None)
    check("⑪ 往返回到 5 帧原尺寸",
          decoded["shape"] == [1, 3, 5, 32, 24], decoded["shape"])
    check("⑪ decode 走的是真 VAE", decoded["vaeKind"] == "seedvr2-real", decoded["vaeKind"])

    msg = _err(lambda: backend.encode_pixels(torch.rand(1, 3, 4, 16, 16)))
    check("⑪ 4 帧 ⇒ 编码前就拒（4n+1）", "4n+1" in msg, msg)
    msg = _err(lambda: backend.encode_pixels(torch.rand(1, 3, 16, 16)))
    check("⑪ 4D 输入 ⇒ 明说先补 T", "5D" in msg, msg)


# --------------------------------------- ⑫ 时间分块（带块外真实上文） ----
def case_temporal_chunk() -> None:
    """时间维分块必须与整段**相等** ✓✗ —— 帧数（``4T-3``）＋ 数值两条判据 ✓，
    再来一条**反向**判据：把块外上文掐掉 ⇒ 数值必须变 ✓（证明逐块缓存是真在起作用 ✓）。"""
    print("⑫ 时间分块：与整段相等 ＋ 反向验证")
    torch.manual_seed(7)
    for temporal_scale_num in (1, 2):
        shape = sv.SeedVR2VaeShape(
            block_out_channels=(8, 16, 16), layers_per_block=1, norm_num_groups=8,
            latent_channels=4, temporal_scale_num=temporal_scale_num,
        )
        vae = sv.build(shape)
        z = torch.randn(1, 4, 13, 6, 6)
        with torch.no_grad():
            whole = vae.decode(z)
        want = 1 + (13 - 1) * shape.temporal_scale
        check(f"⑫ [×{shape.temporal_scale}] 整段帧数 = 4T-3", whole.shape[2] == want,
              (int(whole.shape[2]), want))
        for chunk in (1, 2, 4, 6):
            with torch.no_grad():
                part = vae.decode(z, temporal_chunk=chunk)
            diff = _maxdiff(part, whole) if tuple(part.shape) == tuple(whole.shape) else float("nan")
            check(f"⑫ [×{shape.temporal_scale}] chunk={chunk} 帧数与整段相同",
                  tuple(part.shape) == tuple(whole.shape), (tuple(part.shape), tuple(whole.shape)))
            #: ⚠️ 判据卡 **1e-5** ✗ 不放宽到"看着差不多"✓：块首帧走的是**真实上文** ✓ 还是
            #: "复制首帧" ✓✗，差别是 ``1.8~2.8`` 那个量级 ✓（两轮实测 ✓）—— 拿 bf16/fp32 抖动
            #: 解释不了 ✓。剩下那点 1e-6 来自卷积在不同 ``T`` 上选的核不同 ✓，不是口径错 ✓。
            check(f"⑫ [×{shape.temporal_scale}] chunk={chunk} 与整段逐位（≤1e-5）",
                  diff <= 1e-5, diff)
        #: 反向：块切法、帧数都对 ✓，只是每块拿**空缓存** ⇒ ``ACTIVE`` 退化成"复制首帧" ✓✗
        #: ⇒ 帧数仍对 ✓、数值必须崩 ✓（这条要是也过 ⇒ 说明缓存根本没接上 ✗）。
        head, rest = z[:, :, :1], z[:, :, 1:]
        pieces = []
        for index, piece in enumerate(rest.split(4, dim=2)):
            state = sv._MemoryState.INITIALIZING if index == 0 else sv._MemoryState.ACTIVE
            if index == 0:
                piece = torch.cat((head, piece), dim=2)
            with torch.no_grad():
                pieces.append(vae.decoder(piece, state, {}))
        naive = torch.cat(pieces, dim=2)
        broken = _maxdiff(naive, whole) if tuple(naive.shape) == tuple(whole.shape) else float("inf")
        check(f"⑫ [×{shape.temporal_scale}] 反向：掐掉块外上文 ⇒ 数值崩（缓存真起作用）",
              broken > 1e-2, broken)

        #: ⚠️⚠️ 缓存**不许**是整块激活的视图 ✗✗：``x[:, :, mem_size:].detach()`` 不复制存储 ✓
        #: ⇒ 每层都把这一块的**整份激活**钉住不放 ✓ ⇒ 分块峰值反而高过整段 ✓（实测 720×480 /
        #: chunk=2 是 22.62 GiB vs 整段 13.48 GiB ✓）。判据直接量**存储大小**：只该装 ``mem_size`` 帧 ✓。
        probe = sv._InflatedCausalConv3d(4, 4, kernel_size=3, stride=1, padding=1)
        chunk_in = torch.randn(1, 4, 9, 2, 2)
        cache: dict = {}
        with torch.no_grad():
            probe(chunk_in, sv._MemoryState.INITIALIZING, cache)
        per_frame = chunk_in[0, :, 0].numel() * chunk_in.element_size()
        held = cache[probe].untyped_storage().nbytes() // per_frame
        check(f"⑫ [×{shape.temporal_scale}] 缓存只拷 2 帧、不钉住整块（.detach 视图 ⇒ 9 帧）",
              cache[probe].shape[2] == 2 and held <= 2, (int(cache[probe].shape[2]), int(held)))

    if not REAL.exists():
        skip("⑫ 真权重分块", f"权重未就位：{REAL}")
        return
    vae = _real_vae()
    z = torch.randn(1, 16, 13, 3, 3)
    with torch.no_grad():
        whole = vae.decode(z)
        part = vae.decode(z, temporal_chunk=4)
    check("⑫ 真权重（250 键）：分块帧数与整段相同",
          tuple(part.shape) == tuple(whole.shape), (tuple(part.shape), tuple(whole.shape)))
    diff = _maxdiff(part, whole) if tuple(part.shape) == tuple(whole.shape) else float("nan")
    check("⑫ 真权重：分块与整段逐位（≤1e-4）", diff <= 1e-4, diff)


# --------------------------------------- ⑬ 后端：OOM 只降「时间分块」 ----
def case_oom_ladder() -> None:
    """``fallback`` 在**这一族**降的必须是**时间分块** ✓✗ —— 降 ``tile`` 救不了这份峰值 ✓
    （它来自时间 ×2 上采样后的重排 ✓），而时间分块**不动数值** ✓ ⇒ 这是本仓唯一
    「降档不换画质」的兜底 ✓（Wan 那族降档是**换字节** ✗，两者读回执的方式不同 ✓）。

    四条判据各抓一种坏法 ✓：
    ① 真 OOM ⇒ 降一档重试**一次** ✓、回执 ``reason="oom"`` ＋ ``exact`` ✓；
    ② **非** OOM 的错 ⇒ 原样抛、**不**降档 ✓（盖上"已降档"就是把真 bug 埋掉 ✗✗）；
    ③ ``fallback=False`` ⇒ 真报错 ✓（别把"兜底"当成"永远兜" ✗）；
    ④ 降完仍 OOM ⇒ 人话点明「降帧数/分辨率，别降精度」✓。
    """
    print("⑬ 后端兜底：OOM ⇒ 只降时间分块（不换画质）")
    from app.services.engine import torch_backend as tb

    backend = tb.TorchBackend(device="cpu")
    backend._vae = torch.nn.Identity()
    backend._vaeKind = "seedvr2-real"
    backend._vaeLatents = None
    backend._vaeConfig = None
    latents = torch.zeros(1, 16, 13, 2, 2)
    plan = type("P", (), {"frames": 5})()
    original = sv.decode_latents

    def _run(script, *, tiling=None, fallback=None):
        """``script`` ＝ 每次调用的行为（``"oom"``／``"wrong"``／``"ok"`` 轮着来 ✓）。"""
        calls: list = []
        index = {"i": 0}

        def fake(vae, z, **kwargs):  # noqa: ANN001, ARG001
            calls.append(kwargs.get("temporal_chunk"))
            step = script[min(index["i"], len(script) - 1)]
            index["i"] += 1
            if step == "oom":
                raise RuntimeError("CUDA out of memory. Tried to allocate 17.30 GiB")
            if step == "wrong":
                raise RuntimeError("Expected all tensors to be on the same device")
            return torch.zeros(1, 3, 5, 2, 2)

        if tiling is not None or fallback is not None:
            backend.set_vae_tiling(temporal_chunk=tiling,
                                   fallback=True if fallback is None else fallback)
        else:
            backend.reset_vae_tiling()
        sv.decode_latents = fake
        try:
            return _err(lambda: backend.decode(latents, plan, None)), calls, index
        finally:
            sv.decode_latents = original

    # ① 真 OOM ⇒ 降一档（``SUGGESTED_TEMPORAL_CHUNK`` ✓）重试一次 ✓
    backend.reset_vae_tiling()
    calls: list = []
    out: dict = {}
    index = {"i": 0}

    def oom_then_ok(vae, z, **kwargs):  # noqa: ANN001, ARG001
        calls.append(kwargs.get("temporal_chunk"))
        index["i"] += 1
        if kwargs.get("temporal_chunk") is None:
            raise RuntimeError("CUDA out of memory. Tried to allocate 17.30 GiB")
        return torch.zeros(1, 3, 5, 2, 2)

    sv.decode_latents = oom_then_ok
    try:
        out = backend.decode(latents, plan, None)
    finally:
        sv.decode_latents = original
    note = out.get("vaeTiling") or {}
    check("⑬ 真 OOM ⇒ 降到 SUGGESTED_TEMPORAL_CHUNK（只降一档、只重试一次）",
          calls == [None, sv.SUGGESTED_TEMPORAL_CHUNK]
          and note.get("temporalChunk") == sv.SUGGESTED_TEMPORAL_CHUNK
          and note.get("reason") == "oom", (calls, note))
    check("⑬ 兜底标 exact=True（这一族降档不换画质 ✓，与 Wan 那族要分开读 ✓✗）",
          note.get("exact") is True, note)
    check("⑬ 降档后仍然出片（不是把 OOM 变成失败）",
          out.get("shape") == [1, 3, 5, 2, 2], out.get("shape"))

    # ② 非 OOM ⇒ 原样抛、一次都不重试 ✓
    backend.reset_vae_tiling()
    msg, calls, _ = _run(["wrong"])
    check("⑬ 非 OOM 的错 ⇒ 原样抛（不降档、不重试）",
          "same device" in msg and calls == [None], (msg, calls))

    # ③ fallback=False ⇒ 真报错 ✓
    msg, calls, _ = _run(["oom"], fallback=False)
    check("⑬ `fallback=false` ⇒ OOM 真报错（不许偷偷降档）",
          "out of memory" in msg and calls == [None], (msg, calls))

    # ④ 降完仍 OOM ⇒ 人话 ✓（阶梯要**整条试遍**才对得起「真的超了」这句 ✓）
    msg, calls, _ = _run(["oom", "oom"])
    ladder = list(sv.chunk_ladder())
    check("⑬ 降完仍 OOM ⇒ 人话点明「降帧数/分辨率，别降精度」",
          "temporalChunk" in msg and "别去降精度" in msg
          and calls == [None, *ladder], (msg, calls))
    check("⑬ 阶梯是 4→2→1（逐档减半、1 是地板）",
          sv.chunk_ladder() == (4, 2, 1) and sv.chunk_ladder(3) == (3, 1)
          and sv.chunk_ladder(1) == (1,), (sv.chunk_ladder(), sv.chunk_ladder(3)))

    # ⑥ ⭐ 「一块就顶到上限」那档：4 号仍 OOM、2 号过得去 ⇒ 必须**继续下探**，不是降一档就收工
    def oom_until_two(vae, z, **kwargs):  # noqa: ANN001, ARG001
        calls.append(kwargs.get("temporal_chunk"))
        if kwargs.get("temporal_chunk") in (None, 4):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.83 GiB")
        return torch.zeros(1, 3, 5, 2, 2)

    backend.reset_vae_tiling()
    calls = []
    sv.decode_latents = oom_until_two
    try:
        out = backend.decode(latents, plan, None)
    finally:
        sv.decode_latents = original
    note = out.get("vaeTiling") or {}
    check("⑬ 4 号也 OOM ⇒ 继续下探到 2（阶梯不是「降一档就收工」）",
          calls == [None, 4, 2] and note.get("temporalChunk") == 2, (calls, note))
    check("⑬ 下探后的回执把试过的档记下来",
          note.get("tried") == [4, 2] and note.get("reason") == "oom", note)

    # ⑦ 溢出估算：真数据（chunk=4 峰值 reserved 86.67 GiB / 卡 22.49 GiB ⇒ 该直接降到 1）
    card = int(22.49 * 2**30)
    check("⑬ 溢出 ⇒ 按「峰值÷卡容量」直接跳到该降的档（4 → 1，别一档档慢慢疼）",
          sv.shrink_for_peak(4, int(86.67 * 2**30), card) == 1
          and sv.shrink_for_peak(2, int(38.65 * 2**30), card) == 1, None)
    check("⑬ 没溢出（峰值 ≤ 卡）⇒ 不动档位",
          sv.shrink_for_peak(4, int(10 * 2**30), card) is None, None)
    check("⑬ 已经在 1 档 ⇒ 再降没有意义",
          sv.shrink_for_peak(1, int(86 * 2**30), card) is None, None)

    # ⑧ 上一轮**自动降**下来的档位，回执不能写 ``configured`` ✓✗（会让人以为是他自己配的 ✓）
    backend.reset_vae_tiling()
    backend._vaeTiling["temporalChunk"] = 2
    backend._vaeAutoChunk = 2
    sv.decode_latents = lambda vae, z, **kw: torch.zeros(1, 3, 5, 2, 2)  # noqa: ARG005
    try:
        out = backend.decode(latents, plan, None)
    finally:
        sv.decode_latents = original
        backend.reset_vae_tiling()
    check("⑬ 自动降的档如实标 ``reason=\"auto\"``（别冒充创作者配的）",
          (out.get("vaeTiling") or {}).get("reason") == "auto", out.get("vaeTiling"))

    # ⑤ 显式配了 ⇒ 直接照办、标 configured（这才是「旋钮真的接进解码器」✓）
    out = {}
    sv.decode_latents = lambda vae, z, **kw: (out.update(kw)
                                              or torch.zeros(1, 3, 5, 2, 2))  # noqa: ARG005
    backend.set_vae_tiling(temporal_chunk=2)
    try:
        got = backend.decode(latents, plan, None)
    finally:
        sv.decode_latents = original
        backend.reset_vae_tiling()
    check("⑬ 显式 `temporalChunk=2` ⇒ 原样传到 `decode_latents`",
          out.get("temporal_chunk") == 2, out)
    check("⑬ 显式配置的回执标 configured",
          (got.get("vaeTiling") or {}).get("reason") == "configured", got.get("vaeTiling"))


def main() -> int:
    print("=" * 78)
    print("SEEDVR2 VAE（VideoAutoencoderKL）自检")
    print("=" * 78)
    for case in (case_infer, case_infer_guard, case_shape_problems, case_keys,
                 case_frames, case_small, case_causal, case_golden, case_reverse,
                 case_readout, case_backend, case_temporal_chunk, case_oom_ladder):
        case()
        print()
    if SKIPS:
        print(f"（跳过 {len(SKIPS)} 项 ✓：{'、'.join(SKIPS)}）")
    print(f"SUMMARY: {len(CHECKS) - len(FAILS)}/{len(CHECKS)} passed")
    if FAILS:
        print("FAILED: " + "、".join(FAILS))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
