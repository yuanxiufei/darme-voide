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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from app.services.engine import latent_formats as lf  # noqa: E402
from app.services.engine import safetensors as st  # noqa: E402
from app.services.engine import torch_backend as tb  # noqa: E402
from app.services.engine import vae_wan  # noqa: E402
from app.services.engine.weights import load_module_weights  # noqa: E402

FAILS: list[str] = []

REAL = Path("D:/Comfy-Desktop/ComfyUI-Shared/models/vae/qwen_image_2.1_vae_bf16.safetensors")

#: ★ 最后一级必须 ``in == out`` ✗（``down_flag=False`` 那一级的 ``AvgDown3D`` 是纯通道重排，
#: 参考实现在构造时就 ``assert in*1 % out == 0`` ⇒ ``dim_mult`` 末尾要重复一次 ✓）
SMALL = vae_wan.WanVaeShape(
    dim=8, dec_dim=12, z_dim=4, dim_mult=(1, 2, 2), num_res_blocks=1,
    temperal_downsample=(True, True), image_channels=3, patch_size=1, temporal_kernel=1,
)


def check(name: str, ok: object, detail: object = None) -> None:
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


def main() -> int:
    print("=" * 60)
    print("Wan VAE（Qwen-Image 2.1 · 64ch）自检")
    print("=" * 60)
    case_small()
    case_real()
    print("\n" + "=" * 60)
    if FAILS:
        print(f"失败 {len(FAILS)} 项：")
        for name in FAILS:
            print("  -", name)
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
