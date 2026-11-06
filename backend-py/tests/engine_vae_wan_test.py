"""Wan 系 VAE（Qwen-Image 2.1 的 64 通道那一支）的**结构与数值**自检。

判据五层：

① **形状反推**：从盘上真权重的 238 个键反推出的 ``dim/dec_dim/z_dim/dim_mult/
   temperal_downsample/image_channels/patch_size/temporal_kernel/scale`` 必须逐项正确，
   且 ``scale`` 必须与族事实 ``latent_formats.QwenImage21.spacial_downscale_ratio=16`` 一致。
② **键对账**：``load_module_weights`` 必须 ``complete``（0 缺 0 错形）—— 这一条抓
   「无参层写漏」（``ResidualBlock`` 里那个 ``Dropout`` ✗）、「多建一层」（``time_conv`` ✗）。
③ **数值对拍**：与 ``tests/vae_wan_ref.py``（comfy 原文，只用 ``importlib`` 加载、数学一行未改）
   比 ``decode`` 输出；还要求 ``8×8 → 128×128``（放大 16 倍）。
④ **反向验证**：把三处最容易错的口径各改坏一次（``first_chunk`` 丢了、``DupUp3D`` 用被覆盖
   的 ``in_dim``、``temperal_downsample`` 多留最后一级），断言**装载报缺键或数值显著偏离** ——
   证明③分辨力够，不是「两份都错也一样绿」。
⑤ **小形状自洽**：不依赖盘上文件，随机小配置也能建、也能与参照逐位对齐（CI 无模型时照跑）。
⑥ **反缩放 + 接入**：64 个 ``latents_mean/std`` 与 comfy 原文 ``QwenImage21`` **逐值相同**
   （用 ``ast`` 抠原文，不 import comfy ✓）、``decode_latents`` 的方向对（≠ 忘掉反缩放的结果 ✓）、
   取不到族名**当场抛**（不回退 ``(0,1)`` ✓✗）、``TorchBackend.attach_wan_vae`` 装得上且
   ``decode`` 走对分支并**点名族**（不许写成 H3 ✗）。
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: ⚠️ cp936 控制台下 ``✓`` 会 ``UnicodeEncodeError`` ⇒ 判据还没跑就崩在第一条打印上 ✗
#: （``run_all`` 已重配 stdout，只有单独跑才露头 ⇒ 每个 ``__main__`` 测试都自带这一句 ✓）。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


import torch  # noqa: E402

from app.services.engine import latent_formats as lf  # noqa: E402
from app.services.engine import safetensors as st  # noqa: E402
from app.services.engine import torch_backend as tb  # noqa: E402
from app.services.engine import vae_tiling  # noqa: E402
from app.services.engine import vae_wan  # noqa: E402
from app.services.engine.weights import load_module_weights  # noqa: E402

FAILS: list[str] = []
#: 全部判据名 ✓ —— 收尾要报「过了几项／共几项」✗✗：原来只印「全部通过」✓✗，而
#: `run_all.py` **只认 `SUMMARY: n/m passed`** ✓ ⇒ 本套在全量表里**项数恒为 0**（"凭空消失" ✗）。
CHECKS: list[str] = []

REAL = Path("D:/Comfy-Desktop/ComfyUI-Shared/models/vae/qwen_image_2.1_vae_bf16.safetensors")

#: ★ 最后一级必须 ``in == out`` ✗（``down_flag=False`` 那一级的 ``AvgDown3D`` 是纯通道重排，
#: 参考实现在构造时就 ``assert in*1 % out == 0`` ⇒ ``dim_mult`` 末尾要重复一次 ✓）
SMALL = vae_wan.WanVaeShape(
    dim=8, dec_dim=12, z_dim=4, dim_mult=(1, 2, 2), num_res_blocks=1,
    temperal_downsample=(True, True), image_channels=3, patch_size=1, temporal_kernel=1,
)


def check(name: str, ok: object, detail: object = None) -> None:
    CHECKS.append(name)
    print(f"  {'✓' if ok else '✗'} {name}")
    if not ok:
        FAILS.append(name)
        if detail is not None:
            print(f"      {detail}")


def _shapes(path: Path) -> dict[str, tuple[int, ...]]:
    return {k: tuple(e.shape) for k, e in st.inspect(path).tensors.items()}


# ------------------------------------------------------------------ ⑤ 参照（小形状）
def _reference_small(seed: int = 0) -> tuple[torch.nn.Module, vae_wan.WanVAE]:
    """建一份 comfy 原文的小模型，把它的 state_dict 灌进我的实现 —— 两边同权重 ✓。"""
    from tests.vae_wan_ref import _install_fake_comfy

    mod = _install_fake_comfy()
    torch.manual_seed(seed)
    ref = mod.WanVAE(
        dim=SMALL.dim, dec_dim=SMALL.dec_dim, z_dim=SMALL.z_dim,
        dim_mult=list(SMALL.dim_mult), num_res_blocks=SMALL.num_res_blocks,
        attn_scales=[], temperal_downsample=list(SMALL.temperal_downsample),
        dropout=0.0, image_channels=SMALL.image_channels,
        patch_size=SMALL.patch_size, temporal_kernel=SMALL.temporal_kernel,
    ).double().eval()
    mine = vae_wan.build(SMALL).double().eval()
    report = mine.load_state_dict(ref.state_dict(), strict=False)
    if report.missing_keys or report.unexpected_keys:
        raise RuntimeError(f"小模型键不一致：缺 {report.missing_keys[:4]} 多 {report.unexpected_keys[:4]}")
    return ref, mine


def _diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """相对差（按参照的尺度归一）✓ —— 绝对阈值会被小权重骗过 ✗。"""
    scale = float(b.abs().max()) or 1.0
    return float((a - b).abs().max()) / scale


def case_small() -> None:
    print("\n⑤ 小形状自洽（不读盘 ✓）")
    ref, mine = _reference_small()
    z = torch.randn(1, SMALL.z_dim, 4, 4, dtype=torch.float64)
    with torch.no_grad():
        got, want = mine.decode(z), ref.decode(z)
    check("小模型放大 4×", tuple(got.shape) == (1, 3, 16, 16), tuple(got.shape))
    rel = _diff(got, want)
    check("小模型与参照对齐", rel < 1e-10, f"rel={rel:.3e}")
    check("scale 属性 = 4", SMALL.scale == 4, SMALL.scale)


# ------------------------------------------------------------------ ⑥ 反缩放
#: comfy 原文（**只读来抠数** ✓ —— 不 import ✓：那份文件 ``import comfy.nested_tensor`` ✓✗）
COMFY_LATENT_FORMATS = Path("D:/code/voide/comfy-studio/ComfyUI/comfy/latent_formats.py")


def _comfy_class_stats(class_name: str) -> tuple[list[float], list[float]] | None:
    """``ast`` 抠参考实现某个 ``class`` 里前两个 ``torch.tensor([...])`` ✓（mean 在前 ✓）。

    ⚠️ 抠的是**字面量** ✗ 不是跑出来的张量 ✓：这样连「抄错一位」「少抄一个数」都能抓到 ✓，
    而且不需要 ``comfy`` 包可导入 ✓。⚠️ 只认 ``torch.tensor`` 调用 ✓ —— 同文件里那些
    ``latent_rgb_factors = [[...]]`` 是**裸列表** ✓ 不会被算进来 ✓。
    """
    if not COMFY_LATENT_FORMATS.exists():
        return None
    try:
        tree = ast.parse(COMFY_LATENT_FORMATS.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return None
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        found: list[list[float]] = []
        for inner in ast.walk(node):
            if (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "tensor" and inner.args
                    and isinstance(inner.args[0], ast.List)):
                found.append([float(ast.literal_eval(item)) for item in inner.args[0].elts])
        if len(found) >= 2:
            return found[0], found[1]
    return None


def _comfy_qwen21_stats() -> tuple[list[float], list[float]] | None:
    """``class QwenImage21`` 的那两串 ✓（64 个 ✓）。"""
    return _comfy_class_stats("QwenImage21")


def _raises(action) -> Exception | None:
    try:
        action()
    except Exception as exc:  # noqa: BLE001 —— 任何异常都算「被拦住了」✓
        return exc
    return None


def case_latents(shape: vae_wan.WanVaeShape, mine: vae_wan.WanVAE) -> None:
    print("\n⑥ 反缩放（latents_mean/std）与 torch_backend 接入")
    spec = lf.get_latent_format("qwen_image21")
    check("⑥ 口径登记：64 通道 / 2 维 / 16× / meanstd",
          (spec.latent_channels, spec.latent_dimensions, spec.spatial_downscale_ratio,
           spec.process) == (64, 2, 16, lf.PROCESS_MEANSTD), spec.to_dict())
    check("⑥′ 16× 与 `WanVaeShape.scale` **同源** ✓（4 次下采样 ✓；写 32 就红 ✓）",
          spec.spatial_downscale_ratio == shape.scale == shape.spatial_scale == 16,
          (spec.spatial_downscale_ratio, shape.scale))
    check("⑥″ 通道数与 `z_dim` 同源 ✓（64 ✓ —— 别名属性给 `_check_attached_vaes` 读 ✓）",
          spec.latent_channels == shape.z_dim == shape.latent_channels == 64,
          (spec.latent_channels, shape.z_dim))
    hunyuan = lf.get_latent_format("hunyuan_image_21")
    check("⑥‴ 与 `hunyuan_image_21` **不是**同一条 ✓✗（同为 64 通道 ✓ 但换算两回事 ✓）",
          hunyuan.name != spec.name and hunyuan.process != spec.process,
          (hunyuan.process, spec.process))

    #: ⭐ 2026-10-31：``wan21`` 从「16 个 0/1」修成**真均值/标准差** ✓✗。
    #: 这一格是**视频**视频 VAE 的口径 ✓（16ch / 3 维 / 8× / 时间 4× ✓），与上面 2.1 的
    #: 「64ch / 2 维 / 16×」**每一项都不同** ✓ —— 拿错不会报错 ✓，只是颜色与帧数一起错 ✓✗。
    wan = lf.get_latent_format("wan21")
    check("⑥ᵃ `wan21` 口径：16 通道 / **3 维** / 8× / **时间 4×** / meanstd ✓✗（漏时间那一格 ⇒ "
          "帧数按不下采样算 ⇒ 解出 4 倍帧 ✓✗）",
          (wan.latent_channels, wan.latent_dimensions, wan.spatial_downscale_ratio,
           wan.temporal_downscale_ratio, wan.process)
          == (16, 3, 8, 4, lf.PROCESS_MEANSTD), wan.to_dict())
    wan_mean, wan_std = vae_wan.latents_stats("wan21", z_dim=16)
    check("⑥ᵇ `wan21` 的 mean/std 是**真值** ✓✗（不是 16 个 0/1：0/1 ⇒ 恒等 ⇒ 图整体偏灰、"
          "解码照跑不报错 ✓✗）",
          abs(wan_mean[0] + 0.7571) < 1e-6 and abs(wan_std[0] - 2.8184) < 1e-6
          and wan_mean != (0.0,) * 16, (wan_mean[:3], wan_std[:3]))
    check("⑥ᶜ 2512 那族（`qwen_image`）**仍是恒等** ✓ —— 与 `wan21` 是两族 ✓✗（借名 ⇒ 吃到 Wan 的 "
          "mean/std ⇒ 图偏绿 ✓✗）",
          vae_wan.latents_stats("qwen_image", z_dim=16) == ((0.0,) * 16, (1.0,) * 16)
          and vae_wan.latents_stats("qwen_image", z_dim=16) != (wan_mean, wan_std), "")

    #: ⭐ Wan 2.2 是**另一组 48 个数** ✓（``class Wan22(Wan21)`` 自己重写了 mean/std ✓）——
    #: 拿 Wan21 那 16 个顶、或"随手补 32 个 0/1"，都是解码照跑、颜色偏、一个错都不报 ✓✗。
    wan22_mean, wan22_std = vae_wan.latents_stats("wan22", z_dim=48)
    check("⑥ᵈ `wan22` 是 48 个**独立**的数 ✓✗（不是 Wan21 那 16 个、也不是补 0/1 ✓）",
          len(wan22_mean) == len(wan22_std) == 48
          and abs(wan22_mean[0] + 0.2289) < 1e-6 and abs(wan22_std[0] - 0.4765) < 1e-6
          and wan22_mean != (0.0,) * 48, (wan22_mean[:3], wan22_std[:3]))
    check("⑥ᵉ `wan22` 配 16 通道 ⇒ **报错** ✓（通道守卫不许被 48 个数顶过去 ✓✗；"
          "三个别名 `wan_2_2` / `wan2.2` 都要指到同一组 ✓）",
          _raises(lambda: vae_wan.latents_stats("wan22", z_dim=16)) is not None
          and vae_wan.latents_stats("wan_2_2", z_dim=48) == (wan22_mean, wan22_std)
          and vae_wan.latents_stats("wan2.2", z_dim=48) == (wan22_mean, wan22_std), "")
    comfy22 = _comfy_class_stats("Wan22")
    if comfy22 is None:
        print("      （跳过 Wan22 原文比对：那份文件不在）")
    else:
        check("⑥ᶠ `wan22` 与参考实现 ``class Wan22`` **逐值相同** ✓（抄错一位就红 ✓）",
              list(wan22_mean) == comfy22[0] and list(wan22_std) == comfy22[1],
              (len(comfy22[0]), len(comfy22[1])))

    stats = _comfy_qwen21_stats()
    if stats is None:
        print("      （跳过 comfy 原文比对：那份文件不在）")
    else:
        comfy_mean, comfy_std = stats

        check("⑥⁗ comfy 原文抠到 64 + 64 个数",
              (len(comfy_mean), len(comfy_std)) == (64, 64),
              (len(comfy_mean), len(comfy_std)))
        got_mean, got_std = vae_wan.QWEN21_LATENT_MEAN, vae_wan.QWEN21_LATENT_STD
        mean_diff = max(abs(a - b) for a, b in zip(got_mean, comfy_mean, strict=False))
        std_diff = max(abs(a - b) for a, b in zip(got_std, comfy_std, strict=False))
        check("⑥⁵ mean 逐值相同 ✓（抄错一位就红 ✓）",
              len(got_mean) == len(comfy_mean) and mean_diff == 0.0, mean_diff)
        check("⑥⁶ std 逐值相同 ✓", len(got_std) == len(comfy_std) and std_diff == 0.0, std_diff)

    mean, std = vae_wan.latents_stats("qwen_image21", z_dim=64)
    check("⑥⁷ 取不到族名 ⇒ **报错** ✗（不回退 ``(0,1)`` ✓✗）",
          #: ⚠️ 拿 ``wan22`` 当"没登记"的例子已经过期 ✗ —— 它 2026-10-31 起**登记了真值** ✓。
          _raises(lambda: vae_wan.latents_stats("wan9_nope")) is not None)
    check("⑥⁸ 通道数对不上 ⇒ **报错** ✗（64 个数配 48 通道 ⇒ 不硬套 ✓✗）",
          _raises(lambda: vae_wan.latents_stats("qwen_image21", z_dim=48)) is not None)

    z = torch.randn(1, 64, 4, 4)
    std_t = torch.tensor(std).view(1, -1, 1, 1)
    mean_t = torch.tensor(mean).view(1, -1, 1, 1)
    with torch.no_grad():
        via_helper = vae_wan.decode_latents(mine, z)
        hand_scaled = mine.decode(z * std_t + mean_t)     # 手算反缩放（同样的事 ✓）
        forgot = mine.decode(z)                          # 忘掉反缩放（错的那条 ✓✗）
        unit = vae_wan.decode_latents(mine, z, to_unit=True)
    check("⑥⁹ `decode_latents` == ``decode(z*std+mean)`` ✓（方向对 ✓）",
          _diff(via_helper, hand_scaled) < 1e-5, f"rel={_diff(via_helper, hand_scaled):.3e}")
    check("⑥⁵′ 忘掉反缩放 ⇒ 输出**显著不同** ✓✗（判据有分辨力 ✓）",
          _diff(forgot, hand_scaled) > 0.1, f"rel={_diff(forgot, hand_scaled):.3e}")
    check("⑥⁶′ `to_unit=True` ⇒ 3 通道 / [0,1] ✓",
          tuple(unit.shape) == (1, 3, 64, 64)
          and float(unit.min()) >= 0.0 and float(unit.max()) <= 1.0, tuple(unit.shape))

    backend = tb.TorchBackend(device="cpu")
    if not backend.describe()["available"]:
        print("      （跳过接入：torch 后端不可用）")
        return
    # 2.1 的 DiT 是 19 GiB int8 ⇒ 这里不装主干 ✓，只**塞口径**让 `_check_attached_vaes` 有得核 ✓
    backend._config = {"in_channels": 64, "vae_scale": 16}
    detail = backend.attach_wan_vae(str(REAL), dtype=torch.float32)
    check("⑥⁷′ `attach_wan_vae` 装上 ✓（238 键 complete ✓ / kind 说真话 ✓）",
          detail.get("complete") is True and detail.get("kind") == "wan21-real"
          and detail.get("loadedKeys") == 238, {k: detail.get(k) for k in ("complete", "kind", "loadedKeys")})
    check("⑥⁸′ 反缩放事实被记下 ✓（64 个 mean/std + 族名 ✓）",
          backend._vaeLatents is not None and len(backend._vaeLatents[0]) == 64,
          None if backend._vaeLatents is None else backend._vaeLatents[2])

    # ⚠️ 2026-10-31 真踩到的坑 ✗：建模块时不定精度 ⇒ fp32 建 + bf16 装 ⇒ `copy_` 把权重压回
    #    float32 ✓✗，报告仍写 dtype=bf16 ✓、参数全是 fp32 ✓✗，喂 bf16 潜变量才在**卷积里**
    #    报 `Input type (BFloat16) and bias type (float) should be the same` ✓（报得像输入错了 ✗✗）。
    backend.attach_wan_vae(str(REAL), dtype=torch.bfloat16)
    dtypes = {str(p.dtype) for p in backend._vae.parameters()}
    check("⑥⁹′ 精度**建的时候就定** ✓✗（bf16 装 ⇒ 参数必须全 bf16 ✓；否则静默变 fp32 ✓✗）",
          dtypes == {"torch.bfloat16"}, dtypes)
    backend.attach_wan_vae(str(REAL), dtype=torch.float32)

    out = backend.decode(z, None, None)
    check("⑥⁹′ `decode` 走对分支 ✓（4 维图 / real=True / vaeKind ✓）",
          out.get("shape") == [1, 3, 64, 64] and out.get("synthetic") is False
          and out.get("vaeKind") == "wan21-real", out.get("shape"))
    check("⑥⁹″ head 的**第 4 通道 alpha 已丢** ✓✗（留 4 通道 ⇒ `media.write_image` 落盘才炸 ✓✗）"
          "且 note 如实写了 ✓",
          bool(out.get("frames") is not None) and int(out["frames"].shape[1]) == 3
          and "alpha" in str(out.get("note")), out.get("note"))
    check("⑥⁵″ note **点名 2.1** ✓✗（旧写法「否则 H3」会把 2.1 写成 H3 ✓✗）",
          "2.1" in str(out.get("note")) and "H3" not in str(out.get("note")), out.get("note"))

    wrong = tb.TorchBackend(device="cpu")
    wrong._config = {"in_channels": 16, "vae_scale": 8}
    err = _raises(lambda: wrong.attach_wan_vae(str(REAL), dtype=torch.float32))
    check("⑥⁶″ 挂错族 ⇒ **当场拒** ✓（64ch/16× 挂到 16ch/8× 后端 ✓）",
          err is not None and "对不上" in str(err), type(err).__name__ if err else "没报错")
    check("⑥⁷″ 失败**不留残骸** ✓（`_vae` / `_vaeKind` 一个字没动 ✓）",
          wrong._vae is None and wrong._vaeKind is None)

    bad_family = tb.TorchBackend(device="cpu")
    family_err = _raises(lambda: bad_family.attach_wan_vae(str(REAL), dtype=torch.float32,
                                                           family="wan22"))
    check("⑥⁸″ 反缩放事实缺 ⇒ **当场拒** ✓（不拿 ``(0,1)`` 顶上 ✓✗）",
          family_err is not None and "反缩放" in str(family_err),
          type(family_err).__name__ if family_err else "没报错")


# ------------------------------------------------------------------ ①②③ 真文件
def case_real() -> None:
    if not REAL.exists():
        print("\n①②③④ 跳过：盘上没有 2.1 的 VAE")
        return
    shapes = _shapes(REAL)
    shape = vae_wan.infer_shape(shapes)

    print("\n① 形状反推（真文件 238 键）")
    check("键数 238", len(shapes) == 238, len(shapes))
    check("dim=96", shape.dim == 96, shape.dim)
    check("dec_dim=144", shape.dec_dim == 144, shape.dec_dim)
    check("z_dim=64", shape.z_dim == 64, shape.z_dim)
    check("dim_mult=(1,2,4,8,8)", shape.dim_mult == (1, 2, 4, 8, 8), shape.dim_mult)
    check("num_res_blocks=2", shape.num_res_blocks == 2, shape.num_res_blocks)
    check("temperal_downsample=(F,T,T,T) ★只 4 项",
          shape.temperal_downsample == (False, True, True, True), shape.temperal_downsample)
    check("image_channels=4（RGBA）", shape.image_channels == 4, shape.image_channels)
    check("patch_size=1", shape.patch_size == 1, shape.patch_size)
    check("temporal_kernel=1", shape.temporal_kernel == 1, shape.temporal_kernel)
    check("scale=16（= latent_formats.QwenImage21 ✓）", shape.scale == 16, shape.scale)
    check("每条参数都有证据行", len(shape.evidence) >= 8, len(shape.evidence))

    print("\n② 键对账（真权重装进我的实现）")
    mine = vae_wan.build(shape)
    report = load_module_weights(mine, REAL, dtype=torch.float32)
    check("装载 complete（0 缺 0 错形）", report.complete,
          f"缺 {report.missing[:6]} 错形 {report.shapeMismatch[:3]} err={report.error}")
    model_keys = {name for name, _ in mine.named_parameters()}
    weight_keys = {k for k in shapes if not k.endswith(("_scale", "input_scale", "comfy_quant"))}
    check("模型参数名 = 权重键（逐字）", model_keys == weight_keys,
          f"多 {sorted(model_keys - weight_keys)[:4]} 少 {sorted(weight_keys - model_keys)[:4]}")
    mine.eval()

    case_latents(shape, mine)

    print("\n③ 与 comfy 原文对拍（真权重、8×8 latent）")
    try:
        from tests.vae_wan_ref import load_reference
        ref = load_reference()
    except Exception as exc:  # comfy 仓位不在／原文加载失败
        print(f"      （跳过：{type(exc).__name__}: {str(exc)[:70]}）")
        return
    torch.manual_seed(0)
    z = torch.randn(1, 64, 8, 8)
    with torch.no_grad():
        got, want = mine.decode(z), ref.decode(z)
    check("形状 8×8 → 128×128×4", tuple(got.shape) == (1, 4, 128, 128), tuple(got.shape))
    rel = _diff(got, want)
    check("与参照相对差 < 1e-4", rel < 1e-4, f"rel={rel:.3e}")

    print("\n④ 反向验证（改坏一处必须被抓到）")
    base = rel

    # (a) DupUp3D 忽略 first_chunk ⇒ t 维翻倍（与 main 路径相加时对不上 ⇒ 当场抛 ✓）
    original_dup = vae_wan.DupUp3D.forward
    vae_wan.DupUp3D.forward = lambda self, x, first_chunk=False: original_dup(self, x, False)
    try:
        with torch.no_grad():
            broken = mine.decode(z)
        caught = broken.shape[2:] != got.shape[2:]
        detail = tuple(broken.shape)
    except Exception as exc:
        caught, detail = True, type(exc).__name__
    finally:
        vae_wan.DupUp3D.forward = original_dup
    check("改坏 first_chunk ⇒ 形状对不上/当场抛", caught, detail)

    # (b) DupUp3D 用被循环覆盖的 in_dim（repeats 变 2×）—— ``DupUp3D`` **无参数** ✓
    # ⇒ 装载不会报错 ✗，只能在 ``decode`` 里炸 ⇒ 必须真跑一遍才算验证 ✓
    original_init = vae_wan.UpResidualBlock.__init__

    def broken_init(self, in_dim, out_dim, mult, temperal_upsample, up_flag, temporal_kernel):
        original_init(self, in_dim, out_dim, mult, temperal_upsample, up_flag, temporal_kernel)
        if self.avg_shortcut is not None:
            self.avg_shortcut = vae_wan.DupUp3D(out_dim, out_dim,
                                                factor_t=2 if temperal_upsample else 1, factor_s=2)

    vae_wan.UpResidualBlock.__init__ = broken_init
    try:
        bad_model = vae_wan.build(shape)
        bad_dup = load_module_weights(bad_model, REAL, dtype=torch.float32)
        bad_model.eval()
        caught = None
        try:
            with torch.no_grad():
                bad_model.decode(z)
        except Exception as exc:
            caught = exc
        if caught is None and not bad_dup.complete:
            caught = RuntimeError(bad_dup.error)
    finally:
        vae_wan.UpResidualBlock.__init__ = original_init
    check("改坏 DupUp3D 的 in_channels ⇒ decode/装载当场炸", caught is not None,
          type(caught).__name__ if caught else "没有报错（隐患 ✗）")

    # (c) temperal_downsample 多留最后一级 ⇒ time_conv 错位、装载缺键
    bad = vae_wan.WanVaeShape(**{**shape.__dict__,
                                 "temperal_downsample": tuple(shape.temperal_downsample) + (False,)})
    bad_report = load_module_weights(vae_wan.build(bad), REAL, dtype=torch.float32)
    check("改坏 temperal_downsample ⇒ 装载报缺键", not bad_report.complete, bad_report.missing[:4])

    # (d) 分辨率必须真的变：取中间 latent 20×12 → 320×192
    with torch.no_grad():
        mid = mine.decode(torch.randn(1, 64, 20, 12))
        mid_ref = ref.decode(torch.randn(1, 64, 20, 12) * 0)  # 只为跑通形状
    check("非正方形 latent 也按 16 倍放大", tuple(mid.shape) == (1, 4, 320, 192), tuple(mid.shape))
    check("全零 latent 参照无 NaN", bool(torch.isfinite(mid_ref).all()))


def case_single_frame() -> None:
    """⑦ 单帧解码 + ``temporal_kernel=3``（2026-11-01 Krea2 端到端实测抓的 ✓）。

    ⚠️ 这一段**不是**「再加一个形状」✗：``temporal_kernel=3`` 的族（``qwen_image_vae`` ✓，
    Krea2 用的就是它 ✓）解单帧时，旧写法「``T==1`` 就不补零」会当场
    ``Calculated padded input size per channel: (1 x 66 x 66). Kernel size: (3 x 3 x 3)`` ✓，
    报错说"核比输入大" ✓、读起来像尺寸算错 ✗ —— 真因是 comfy 在 ``T==1`` 时走
    ``autopad="causal_zero"`` ✓（把时间核**截到最后一拍** ✓，``comfy/ops.py:615`` ✓）。
    """
    print("\n⑦ 单帧解码（T=1）与 `temporal_kernel=3`")
    torch.manual_seed(3)
    conv = vae_wan.CausalConv3d(3, 4, kernel_size=(3, 3, 3), padding=(1, 1, 1)).double().eval()
    x1 = torch.randn(1, 3, 1, 5, 5, dtype=torch.float64)
    with torch.no_grad():
        got = conv(x1)
        want = torch.nn.functional.conv3d(x1, conv.weight[:, :, -1:], conv.bias,
                                          conv.stride, (0, 1, 1), conv.dilation, conv.groups)
        padded = torch.nn.functional.conv3d(
            torch.nn.functional.pad(x1, [0, 0, 0, 0, 2, 0]), conv.weight, conv.bias,
            conv.stride, (0, 1, 1), conv.dilation, conv.groups)
    check("单帧：时间核截到最后一拍（= comfy `causal_zero`）",
          got.shape == want.shape and _diff(got, want) < 1e-12,
          f"{tuple(got.shape)} rel={_diff(got, want):.3e}")
    check("单帧：截核 ≡ 左边补 2 帧零（两条路必须同值，否则时间口径错）",
          _diff(got, padded) < 1e-12, f"rel={_diff(got, padded):.3e}")

    x3 = torch.randn(1, 3, 3, 5, 5, dtype=torch.float64)
    with torch.no_grad():
        got3 = conv(x3)
        want3 = torch.nn.functional.conv3d(
            torch.nn.functional.pad(x3, [0, 0, 0, 0, 2, 0]), conv.weight, conv.bias,
            conv.stride, (0, 1, 1), conv.dilation, conv.groups)
    check("三帧：左边补 2 帧零、帧数不变（T==1 的特例不许污染多帧路径）",
          got3.shape[2] == 3 and _diff(got3, want3) < 1e-12,
          f"{tuple(got3.shape)} rel={_diff(got3, want3):.3e}")

    shape = vae_wan.WanVaeShape(
        dim=8, dec_dim=12, z_dim=4, dim_mult=(1, 2, 2), num_res_blocks=1,
        temperal_downsample=(True, True), image_channels=3, patch_size=1, temporal_kernel=3,
    )
    model = vae_wan.build(shape).double().eval()
    z = torch.randn(1, shape.z_dim, 4, 4, dtype=torch.float64)
    with torch.no_grad():
        out = model.decode(z)
    check("`temporal_kernel=3` 的 VAE 能解单帧（旧写法在这里抛 kernel(3) > input(1)）",
          tuple(out.shape) == (1, 3, 16, 16), tuple(out.shape))
    check("单帧解码输出全有限", bool(torch.isfinite(out).all()))


def case_family_key() -> None:
    """⑧ 族名 → 查表键：`-` / `.` / 空格 / 大小写都得折成同一种写法。

    ⚠️ 这条盯的是**死配置** ✗（2026-11-01 审出来的 ✓）：本模块原先只把 ``-`` 折成 ``_`` ✓
    ⇒ ① 表里登记的 ``wan2.1`` 永远命中不了 ✓✗（写了等于没写 ✓）；② ``qwen-image-2.1``
    这种「连字符 ＋ 点号」组合谁都不命中 ✓✗ ⇒ 回执里回显一串机器名 ✓。两项都不是「错图」级 ✓，
    但都是「表与折法不是同一套口径」✓✗ —— 想登记一种写法，折法就得认它 ✓。
    """
    print("\n⑧ 族名归一（`-` / `.` / 空格 都折成 `_`）")
    spellings = ["qwen_image21", "qwen-image-2.1", "Qwen_Image_2_1", "  QWEN-IMAGE-2.1  "]
    check("四种写法都读成「Qwen-Image 2.1」（大小写 / `-` / `.` / 空格都得折）",
          all(vae_wan.family_title(name) == "Qwen-Image 2.1" for name in spellings),
          [(name, vae_wan.family_title(name)) for name in spellings])
    check("`wan2.1` / `Wan 2.2` 也命中（这两条键原来永远是死配置）",
          vae_wan.family_title("wan2.1") == "Wan 2.1"
          and vae_wan.family_title("Wan 2.2") == "Wan 2.2",
          (vae_wan.family_title("wan2.1"), vae_wan.family_title("Wan 2.2")))
    check("未登记的族**原样回显**、不猜（`h3` / `Krea2` / 空串）",
          vae_wan.family_title("h3") == "h3" and vae_wan.family_title("Krea2") == "Krea2"
          and vae_wan.family_title("") == "",
          (vae_wan.family_title("h3"), vae_wan.family_title("Krea2"), vae_wan.family_title("")))
    check("`latents_stats` 走同一套折法（`qwen-image-2.1` ⇒ 64 组、`wan2.1` ⇒ 16 组）",
          len(vae_wan.latents_stats("qwen-image-2.1")[0]) == 64
          and len(vae_wan.latents_stats("wan2.1")[0]) == 16,
          (len(vae_wan.latents_stats("qwen-image-2.1")[0]),
           len(vae_wan.latents_stats("wan2.1")[0])))
    bad = _raises(lambda: vae_wan.latents_stats("krea9_nope"))
    check("未知族仍**当场报错**（归一化不许把未知族折成某一行）",
          bad is not None and "krea9_nope" in str(bad), type(bad).__name__ if bad else "没报错")


def case_no_spatial_tiling() -> None:
    """⑨ **这一族不能做空间分块解码** —— 反向钉住 2026-11-02 的实测结论 ✓。

    有人一定会想「像素数是平方级 ✓ ⇒ 像 ``VAEDecodeTiled`` 那样切块解 ✓」✗ —— 这一族**不行** ✗：
    解码器 ``middle.1`` 是**全图单头自注意力** ✓（``AttentionBlock`` ✓），潜变量一切块，
    注意力就只看得到自己那块 ✓ ⇒ 拼回来的**整张**都变了 ✓✗，不是接缝问题 ✓。

    实测两组（真 2.1 权重 238 键 ✓ / 小形状同结论 ✓）：
    10×10 潜变量按 6×6 切 ⇒ 相对差 **1.35** ✓；按 16×16 那一档（=整张）⇒ 差 **0** ✓
    （那只是因为**没切** ✓，反证"一分块就变" ✓）。ComfyUI 的 ``WanVAE`` 同样**没有**
    ``decode_tiled`` ✓ —— 这不是本仓偷懒 ✓。

    真要省显存只能切「注意力**之后**的那几级 up 块」✓（后面纯局部卷积 ✓），
    本仓**已做** ✓（``decode(tile=…)`` ✓，见 ⑩ ✓）。
    ⚠️ 本节只钉「按 latent 整段切」这条路不行 ✓✗；上一轮这句曾写「整张解峰值只有几百 MB、
    用不着」✗✗ ⇒ 实测 2512 的 8× VAE 在 2048×1536 上就要 **20.96 GiB** ✓（见 ⑩ 的表 ✓）。
    """
    print("\n⑨ 不做空间分块解码（全图注意力 ⇒ 实测会改整张，反向钉住）")
    shape = vae_wan.WanVaeShape(
        dim=8, dec_dim=12, z_dim=16, dim_mult=(1, 2, 2), num_res_blocks=1,
        temperal_downsample=(True, True), image_channels=3, patch_size=1, temporal_kernel=3,
    )
    model = vae_wan.build(shape).double().eval()
    attn = [n for n, m in model.named_modules() if type(m).__name__ == "AttentionBlock"]
    check("解码器里有**全图**注意力（`decoder.middle`）⇒ 切块会改变整张",
          any(n.startswith("decoder.middle") for n in attn), attn[:4])

    torch.manual_seed(11)
    z = torch.randn(1, 16, 12, 12, dtype=torch.float64)
    mean, std = vae_wan.latents_stats("wan2.1")
    #: ⚠️ 统计量必须是 ``(1,C,1,1)`` ✓✗：一维 ``[C]`` 广播到 ``[B,C,H,W]`` 会撞**维度 3** ✓
    #: （`vae_wan.decode_latents` 里那段注释说的就是这个坑 ✓）。
    view = (1, -1, 1, 1)
    mean_t = torch.tensor(list(mean), dtype=torch.float64).view(view)
    std_t = torch.tensor(list(std), dtype=torch.float64).view(view)

    scale = int(shape.spatial_scale)
    with torch.no_grad():
        whole = vae_wan.decode_latents(model, z, family="wan2.1")
        manual = model.decode(lf.process_out("wan2.1", z, mean=mean_t, std=std_t))
        scaled = lf.process_out("wan2.1", z, mean=mean_t, std=std_t)
        tiled = vae_tiling.decode_latents_tiled(
            model.decode, scaled, spatial_scale=scale,
            tile=5 * scale, overlap=2 * scale, out_channels=3)

    check("默认路径 = 原路 `decode(反缩放(z))`，逐位相同（没被悄悄接上分块）",
          torch.equal(whole, manual), f"maxAbs={float((whole - manual).abs().max()):.3e}")
    check("分块与整张同形（12×12 潜 ⇒ 48×48 像素）",
          tuple(tiled.shape) == tuple(whole.shape) == (1, 3, 48, 48),
          f"{tuple(tiled.shape)} vs {tuple(whole.shape)}")
    rel = _diff(tiled, whole)
    check("按 latent 整段切 vs 整张 **远大于浮点噪声**（实测 ≫ 1e-2）⇒ 这条路不能接",
          rel > 1e-2, f"rel={rel:.3e}")


def case_tiling_knobs() -> None:
    """⑩ 解码省显存的两个旋钮：默认关 ✓、开了会改数值 ✓、halo 不许信调用方 ✓。

    ⚠️ 2026-11-02 实测纠正上一轮的结论 ✗✗：原写「整张解峰值几百 MB ⇒ 分块用不着」✓✗，
    真值（bf16、单帧、卡 22.49 GiB ✓）—— 2512 的 8× VAE 在 **现行主力画布** 2048×1536 上
    就要 **20.96 GiB**（93% 卡 ✓）、3072×2304 直接 OOM ✓。所以这不是「将来的 4K」问题 ✓。

    并且两个旋钮**各治一半** ✓✗（只开一个都救不回来 ✓）：

    | VAE | 画布 | 整张 | ``attn_chunk=1024`` | ``＋tile=512`` |
    |---|---|---|---|---|
    | 2.1 16× | 2048×1536 | 8.69 | 8.69（**没用** ✗） | **3.21** |
    | 2512 8× | 2048×1536 | 20.96 | **3.94** | **1.03** |

    ⚠️ 两条都**不是逐位** ✓✗：``tile`` 相对差实测 5.8e-03（fp32、2048×1536 ✓），且**加 halo
    消不掉**（halo 16→24 仍是 5e-2 ✗ —— 那是 cuDNN 对不同窗口尺寸选不同算法 ✓，不是接缝 ✓）；
    ``attn_chunk`` 在 2.1 上 4096 恰好逐位、1024 就差 8.55e-03 ✓（SDPA 内部自己也切块 ✓）。
    ⇒ 本套钉的是：① 默认路径**逐位**不变 ✓；② 开了之后**差得有上界** ✓（不是「随便差」✓）；
    ③ ``halo`` 给 0 会被抬到感受野下限 ✓（与不传逐位相同 ✓—— 不许信调用方 ✓）。
    """
    print("\n⑩ 解码省显存旋钮（默认关 / 开了改数值 / halo 不信调用方）")
    shape = vae_wan.WanVaeShape(
        dim=8, dec_dim=12, z_dim=16, dim_mult=(1, 2, 2), num_res_blocks=1,
        temperal_downsample=(True, True), image_channels=3, patch_size=1, temporal_kernel=3,
    )
    model = vae_wan.build(shape).double().eval()
    torch.manual_seed(13)
    z = torch.randn(1, 16, 8, 8, dtype=torch.float64)
    scale = int(shape.spatial_scale)
    receptive = model.decoder.receptive_latent()
    check("感受野解析值 ≥ 4（每级 3×3＋上采样都算进去，不是拍脑袋给个数）",
          receptive >= 4, f"receptive_latent={receptive}")

    with torch.no_grad():
        base = model.decode(z)
        again = model.decode(z, tile=None, halo=None, attn_chunk=None)
        tiled = model.decode(z, tile=4 * scale)
        zero_halo = model.decode(z, tile=4 * scale, halo=0)
    check("默认参数 ≡ 显式全关，逐位相同（默认不许悄悄走分块）",
          torch.equal(base, again), f"maxAbs={float((base - again).abs().max()):.3e}")
    check("`tile` 分块与整张同形（8×8 潜 @4× ⇒ 32×32）",
          tuple(tiled.shape) == tuple(base.shape) == (1, 3, 32, 32),
          f"{tuple(tiled.shape)} vs {tuple(base.shape)}")
    rel = _diff(tiled, base)
    check("`tile` 分块相对差 < 1e-2（不是逐位，但不许离谱）", rel < 1e-2, f"rel={rel:.3e}")
    check("`halo=0` 被抬到感受野下限 ⇒ 与不传 halo 逐位相同（调用方说了不算）",
          torch.equal(tiled, zero_halo), f"maxAbs={float((tiled - zero_halo).abs().max()):.3e}")

    #: 注意力分块必须**真的**分块（否则参数写了等于没写 ✗）：spy 掉 SDPA 数调用次数 ✓。
    #: ⚠️ 要数的是 SDPA 调用 ✗✗、不是 `_attend` 调用 ✓ —— `middle` 里只有一个注意力块 ✓，
    #: 所以 `_attend` 每次 decode 只被调 1 次 ✓（分块发生在它**内部** ✓）。
    seen: list = []
    original_sdpa = torch.nn.functional.scaled_dot_product_attention

    def spy(*args, **kwargs):
        seen.append(args[0].shape[2])
        return original_sdpa(*args, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = spy
    try:
        with torch.no_grad():
            big = model.decode(z, attn_chunk=16)
            split_calls = len(seen)
            seen.clear()
            same = model.decode(z, attn_chunk=64)       # n=64 ⇒ chunk>=n ⇒ 整量 ✓
            whole_calls = len(seen)
            seen.clear()
            whole = model.decode(z)
            default_calls = len(seen)
    finally:
        torch.nn.functional.scaled_dot_product_attention = original_sdpa
    check("`attn_chunk=16` 真的把 n=64 切成 4 批（参数写了必须生效）",
          split_calls == 4 and whole_calls == 1 and default_calls == 1,
          (split_calls, whole_calls, default_calls))
    check("`attn_chunk >= n` 退回整量 ⇒ 逐位等于默认路径",
          torch.equal(same, whole), f"maxAbs={float((same - whole).abs().max()):.3e}")
    check("注意力分块相对差 < 1e-2", _diff(big, whole) < 1e-2, f"rel={_diff(big, whole):.3e}")

    #: pixel ↔ latent 换算与 patch 对齐（`patch_size>1` 时块边界必须落在 patch 网格上 ✓）。
    padded = vae_wan.WanVaeShape(
        dim=8, dec_dim=12, z_dim=16, dim_mult=(1, 2, 2), num_res_blocks=1,
        temperal_downsample=(True, True), image_channels=3, patch_size=2, temporal_kernel=3,
    )
    other = vae_wan.build(padded).double().eval()
    tile_latent, halo_latent = other._tile_in_latent(4 * scale + 1, 3 * scale + 1)
    check("像素口径 → latent 口径并按 patch 对齐（scale=4 / patch=2 ⇒ 都是偶数）",
          tile_latent % 2 == 0 and halo_latent % 2 == 0 and tile_latent >= 1,
          (tile_latent, halo_latent))
    check("`tile=None` 时换算结果也是 None（不给就别算）",
          other._tile_in_latent(None, None) == (None, None))


# ------------------------------------------------- ⑪ 后端接线（旋钮 / OOM 降档）
def case_backend_wiring() -> None:
    """⑪ 旋钮接到 ``TorchBackend.decode`` 之后 **还认不认真** ✓✗。

    这里钉的不是「能不能解码」✓（那是 ⑥ 的事 ✓），而是三条**只有接线才会错**的事实 ✓：

    ① **默认全关** ✓：装得下的时候不许预先分块 ✗（2048×1536 那份实测要 20.96 GiB / 卡 22.49 ✓
       ⇒ 差一点就"顺手分块省事" ✓✗，而那会换掉整张图的字节 ✓）；
    ② ``fallback`` 只在**真 OOM** 时降档 ✓，且回执 ``vaeTiling.reason="oom"`` 必须写 ✓✗——
       不写 ⇒ 同一 seed 复现不出同一张图时，无从判断是「分块了」还是「模型变了」✓✗；
    ③ **非 OOM 的错必须原样抛** ✗✗：``RuntimeError("size mismatch")`` 降档也救不了 ✓，
       盖上"已降档"就是把真 bug 埋进"显存不够"里 ✓（这套用假 ``decode_latents`` 直接喂错 ✓）。
    """
    print("\n⑪ 后端接线（默认全关 / OOM 才降档 / 非 OOM 照抛）")
    check("`_is_oom` 认两种写法（类型名 + 字串），普通错不认",
          tb._is_oom(type("OutOfMemoryError", (RuntimeError,), {})("x"))
          and tb._is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
          and tb._is_oom(RuntimeError("CUDA error: out of memory"))
          and not tb._is_oom(RuntimeError("size mismatch: expected 64 channels")))
    backend = tb.TorchBackend(device="cpu")
    if not backend.describe()["available"]:
        print("      （跳过：torch 后端不可用）")
        return
    fresh = dict(backend._vaeTiling)
    check("装配默认：旋钮全关（含 SEEDVR2 的时间档 ✓）✓、OOM 兜底开 ✓"
          "（兜底只在「本来要崩」那条路生效 ⇒ 不动逐位 ✓）",
          fresh == {"tile": None, "halo": None, "attnChunk": None,
                    "temporalChunk": None, "fallback": True}, fresh)
    knobs = backend.set_vae_tiling(tile=0, halo=-3, attn_chunk=0)
    check("`0` / 负数 ＝ **关**（不是「0 像素一块」）", knobs["tile"] is None
          and knobs["halo"] is None and knobs["attnChunk"] is None, knobs)
    check("`reset_vae_tiling` 回到装配默认（旋钮全关 ＋ 兜底仍开）",
          backend.reset_vae_tiling() == {"tile": None, "halo": None, "attnChunk": None,
                                         "temporalChunk": None, "fallback": True})

    shape = vae_wan.WanVaeShape(
        dim=8, dec_dim=12, z_dim=16, dim_mult=(1, 2, 2), num_res_blocks=1,
        temperal_downsample=(True, True), image_channels=3, patch_size=1, temporal_kernel=3,
    )
    model = vae_wan.build(shape).double().eval()
    backend._vae = model
    backend._vaeKind = "wan21-real"
    backend._vaeLatents = ((0.0,) * 16, (1.0,) * 16, "wan21")
    backend._vaeConfig = None
    torch.manual_seed(17)
    z = torch.randn(1, 16, 4, 4, dtype=torch.float64)

    original = vae_wan.decode_latents
    calls: list[dict] = []

    def fake(module, latents, **kw):
        calls.append(dict(kw))
        return original(module, latents, **kw)

    vae_wan.decode_latents = fake
    try:
        plain = backend.decode(z, None, None)
    finally:
        vae_wan.decode_latents = original
    check("装得下 ⇒ 全关走旧路 ✓，且回执 `vaeTiling` 是 None（不许假装开了什么）",
          plain.get("vaeTiling") is None and calls and calls[0].get("tile") is None
          and calls[0].get("attn_chunk") is None, plain.get("vaeTiling"))

    seq = {"n": 0}

    def oom_then_ok(module, latents, **kw):
        calls.append(dict(kw))
        seq["n"] += 1
        if seq["n"] == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return original(module, latents, **kw)

    calls.clear()
    vae_wan.decode_latents = oom_then_ok
    try:
        rescued = backend.decode(z, None, None)
    finally:
        vae_wan.decode_latents = original
    note = rescued.get("vaeTiling") or {}
    check("OOM ⇒ 降一档重试一次 ✓（tier=S 建议值），回执 `reason=oom` ✓✗",
          seq["n"] == 2 and note.get("reason") == "oom"
          and note.get("tile") == vae_wan.SUGGESTED_TILE
          and note.get("attnChunk") == vae_wan.SUGGESTED_ATTN_CHUNK, (seq["n"], note))
    check("降档后仍出真画面 ✓（不是空帧、「降档成功」其实啥也没解 ✗）",
          tuple(rescued.get("shape") or ()) == (1, 3, 16, 16) and rescued.get("synthetic") is False,
          rescued.get("shape"))

    backend.set_vae_tiling(tile=8, attn_chunk=16)
    calls.clear()
    vae_wan.decode_latents = fake
    try:
        configured = backend.decode(z, None, None)
    finally:
        vae_wan.decode_latents = original
    check("显式开旋钮 ⇒ **真传下去** ✓（参数写了必须生效，不许像 `--only` 那样被静默吞掉 ✗✗）"
          "＋回执 `reason=configured` ✓",
          calls and calls[0].get("tile") == 8 and calls[0].get("attn_chunk") == 16
          and (configured.get("vaeTiling") or {}).get("reason") == "configured",
          (calls[0] if calls else None, configured.get("vaeTiling")))

    backend.reset_vae_tiling()
    backend.set_vae_tiling(fallback=False)
    seq["n"] = 0
    vae_wan.decode_latents = oom_then_ok
    try:
        err = _raises(lambda: backend.decode(z, None, None))
    finally:
        vae_wan.decode_latents = original
    check("`fallback=False` ⇒ OOM **原样抛** ✓（关掉兜底就得真报错，不许偷偷降档 ✗）",
          err is not None and "out of memory" in str(err).lower() and seq["n"] == 1,
          (type(err).__name__ if err else "没报错", seq["n"]))

    backend.reset_vae_tiling()
    seq["n"] = 0

    def bad(module, latents, **kw):
        calls.append(dict(kw))
        seq["n"] += 1
        raise RuntimeError("size mismatch: expected 64 channels")

    vae_wan.decode_latents = bad
    try:
        err = _raises(lambda: backend.decode(z, None, None))
    finally:
        vae_wan.decode_latents = original
    check("非 OOM 的错 ⇒ **不降档、原样抛** ✓✗（降档救不了它，盖「已降档」就是埋 bug ✗）",
          err is not None and "size mismatch" in str(err) and seq["n"] == 1,
          (type(err).__name__ if err else "没报错", seq["n"]))

    seq["n"] = 0

    def always_oom(module, latents, **kw):
        seq["n"] += 1
        raise RuntimeError("CUDA out of memory. Tried to allocate 4.00 GiB")

    vae_wan.decode_latents = always_oom
    try:
        err = _raises(lambda: backend.decode(z, None, None))
    finally:
        vae_wan.decode_latents = original
    check("降到底仍 OOM ⇒ 报成**可读的一句** ✓（点名两个档位 ＋ 指路降分辨率 ✓；不是原始 aten 串 ✗）",
          err is not None and "仍 OOM" in str(err) and "降分辨率" in str(err)
          and seq["n"] == 2, (type(err).__name__ if err else "没报错", seq["n"]))


# ---------------------------------------------------- ⑥ 时间维（真跑 Wan 2.2 I2V 撞出来的）
#: ``temporal_kernel=3`` ＋ 两级时间下采样 ＝ 真 Wan2.1 的口径 ✓（4 倍时间压缩 ＋ 1 ✓）。
TEMPORAL = vae_wan.WanVaeShape(
    dim=8, dec_dim=12, z_dim=4, dim_mult=(1, 2, 2), num_res_blocks=1,
    temperal_downsample=(True, True), image_channels=3, patch_size=1, temporal_kernel=3,
)


def case_temporal() -> None:
    """``downsample3d`` / ``upsample3d`` **必须真跑 ``time_conv``** ✓。

    2026-11-02 真跑 Wan 2.2 I2V 才露头 ✓：旧 ``Resample.forward`` 从头到尾没调过
    ``time_conv`` ✓ ⇒ 3D 模式退化成纯空间采样 ✓、**时间维一动不动且不报错** ✓✗。
    症状是 ``encode`` 段 ``参考 latent 时间维 17 ✗ ≠ 潜变量 5 ✓``（读起来像"帧数填错了" ✗），
    出片侧则是 **8 秒的片解出 34 帧但帧间不连续** ✗。
    """
    model = vae_wan.build(TEMPORAL)
    model.eval()
    with torch.no_grad():
        z17 = model.encode(torch.zeros(1, 3, 17, 32, 32))
        z9 = model.encode(torch.zeros(1, 3, 9, 32, 32))
        z1 = model.encode(torch.zeros(1, 3, 1, 32, 32))
        y5 = model.decode(torch.zeros(1, TEMPORAL.z_dim, 5, 4, 4))
        y1 = model.decode(torch.zeros(1, TEMPORAL.z_dim, 1, 4, 4))
    check("⭐ `encode(17 帧)` ⇒ **5 潜帧**（＝4 倍时间压缩；旧实现给 17 帧 ✗）",
          tuple(z17.shape) == (1, TEMPORAL.z_dim, 5, 8, 8), tuple(z17.shape))
    check("`encode(17 帧)` 的潜帧数 **不等于** 像素帧数（反向钉住「没压缩」这个错法）",
          z17.shape[2] != 17, tuple(z17.shape))
    check("`encode(9 帧)` ⇒ 3 潜帧（1＋4＋4 的 4k+1 口径 ✓）",
          tuple(z9.shape) == (1, TEMPORAL.z_dim, 3, 8, 8), tuple(z9.shape))
    check("``encode(1 帧)`` 仍是 1 潜帧（单帧路径逐位不变 ✓）",
          tuple(z1.shape) == (1, TEMPORAL.z_dim, 1, 8, 8), tuple(z1.shape))
    check("⭐ `decode(5 潜帧)` ⇒ **17 像素帧**（＝4(T−1)+1；旧实现在这给 5 帧 ✗）",
          tuple(y5.shape) == (1, 3, 17, 16, 16), tuple(y5.shape))
    check("`decode(1 潜帧)` 仍是 1 像素帧（单图解码逐位不变 ✓）",
          tuple(y1.shape) == (1, 3, 16, 16), tuple(y1.shape))
    check("两级时间上采样 ⇒ 帧数是 ``4(T−1)+1`` 而不是 ``4T``（首帧不过时间卷积 ✓）",
          4 * (5 - 1) + 1 == y5.shape[2] != 4 * 5, (y5.shape[2], 4 * 5))

    #: 反向：**两条 3D 分支里都必须真出现 `self.time_conv(...)`** ✓ ——
    #: 旧实现的两条分支都只调 `self.resample` ✓✗，靠"形状刚好对上"是查不出来的 ✓。
    src = inspect.getsource(vae_wan.Resample.forward)
    down_part, up_part = src.split('if self.mode == "upsample3d":')
    check("反向：`downsample3d` 分支里出现 `self.time_conv(` ✓（旧实现没有 ✗）",
          "self.time_conv(" in down_part, True)
    check("反向：`upsample3d` 分支里出现 `self.time_conv(` ✓（旧实现没有 ✗）",
          "self.time_conv(" in up_part, True)
    check("反向：`_duplicate` 真把帧数翻倍 ✓", tuple(vae_wan.Resample._duplicate(
        torch.zeros(1, 2 * TEMPORAL.z_dim, 3, 4, 4)).shape) == (1, TEMPORAL.z_dim, 6, 4, 4),
        tuple(vae_wan.Resample._duplicate(torch.zeros(1, 2 * TEMPORAL.z_dim, 3, 4, 4)).shape))


def main() -> int:
    print("=" * 60)
    print("Wan VAE（Qwen-Image 2.1 · 64ch）自检")
    print("=" * 60)
    case_small()
    case_single_frame()
    case_family_key()
    case_no_spatial_tiling()
    case_tiling_knobs()
    case_backend_wiring()
    case_temporal()
    case_real()
    print("\n" + "=" * 60)
    if FAILS:
        print(f"失败 {len(FAILS)} 项：")
        for name in FAILS:
            #: ⚠️ 前缀必须是 `FAIL` ✓（不是 `  -` ✗）：`run_all` 只把 `FAIL` 开头的行转贴到总表 ✓，
            #: 别的写法在大清单里等于**没写**（红在哪还得重跑那一套才知道 ✗）。
            print(f"FAIL  {name}")
    #: ⚠️ 口径是硬要求 ✗✗（`run_all.py` 抬头那条 ✓）：只认 `SUMMARY: n/m passed` ✓ ——
    #: 原来收尾只印「全部通过」✗ ⇒ 全量表里这套的**项数恒为 0** ✓✗。
    print(f"SUMMARY: {len(CHECKS) - len(FAILS)}/{len(CHECKS)} passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    #: ⚠️ cp936 控制台下判据里的 ``✓`` ⇒ `print` 直接 UnicodeEncodeError ✓✗、0 条判据就退 1 ✓
    #: （同 `engine_win_io_test` 那个只在单独跑时露头的坑 ✓）；`run_all` 下不露头是因为它已重配 stdout ✓。
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
