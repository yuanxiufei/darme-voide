"""Wan2.2 I2V DiT 骨架自检（``dit_wan22.py``）✓ —— 判族 ✅ / 建壳 ✅ / 前向 ✅ / 口径 ✅。

十一条判据（每条都能**单独**失败 ✓）：

① **真权重逐键对账**：``wan2.2_i2v_high_noise_14B_fp8_scaled`` 头里 1908 条 = 真权重 **1095**
   + fp8 伴生 **813**（``.scale_weight`` 406 + ``.scale_input`` 406 + ``scaled_fp8`` 哨兵 1 ✓）；
   几何 40 块 / 5120 / 13824 / 40 头 / 128 · 三轴 44·42·42 / 36 / 16 / 4096 / 256 / (1,2,2) ✓；
   参数量 = 头里逐张量 numel 之和 = **14 288 901 184** ✓。盘上没这文件 ⇒ **跳过**（不算失败 ✓）。
② **形状 → 模块 → 装载**闭环：``state_dict`` 反推 → ``infer_shape`` → ``build`` →
   ``load_state_dict(strict=True)`` 无缺无余 ✓。
③ **前向与独立参照双精度一致**（参照**刻意换写法**：手写 gelu-tanh / 显式 softmax /
   逐轴角度直构 cos·sin / ``e`` 按 ``[b][i]`` 逐路切 ✓）⇒ 相对误差 < 1e-9。
   这一条同时钉住三件**不报错只是图不对**的事：``unpatchify`` 通道在**最内层**（``(p,q,r,c)``
   而非 ``(c,p,q,r)`` ✗）、六路调制的位置（自注意力 0·1·2、FFN 3·4·5）、**交叉注意力不受门控** ✓。
④ **门控的 ``1 +`` 项**：modulation 置零 ⇒ 退化成**无门控**，而不是整支乘 0 ✓✗；
   反向验「把 ``1+e`` 写成 ``e``」的参照必须对不上 ✓。
⑤ **3D RoPE**：1×1×1 网格恰是恒等（位置号从 0 起、不居中 ✓）；三轴 (44,42,42)；旋转保 2 分量
   范数；只作用 q/k（交叉注意力 ``rope=False`` ✓）；``head_dim=2`` ⇒ 出现 0 宽轴 ⇒ 当场抛 ✓。
⑥ **``heads`` 只许查表**：(5120,13824,40) ⇒ 40 ✓；表外几何必须显式给（不给就具名抛，不猜
   ``dim//128`` ✗）。
⑦ **严格性**：latent 4 维 / 通道 16（只喂噪声）/ context 宽错 / t·h·w 非 patch 倍数（**不补边** ✓）
   / sigma 个数不符 ⇒ 五条都具名抛 ✓；bf16 进 bf16 出 ✓。
⑧ **σ ⇒ 时刻的 ×1000 在骨架里**（参考实现放在采样器 ``ModelSamplingDiscreteFlow`` 里 ✓）：
   spy 抓 ``time_embedding`` 输入，必须等于 ``sin(freq_dim, 1000σ)`` ✓、
   且**不**等于 ``sin(freq_dim, σ)`` 那版 ✓。
⑨ **36 通道布局**（噪声 16 + 条件图 16 + 掩码 4）：``split_i2v`` 三段宽度、拼回逐字节相等 ✓。
⑩ **``param_count`` 公式**：等于 ``state_dict`` 逐张量相加 ✓，且**每加一层的增量**等于该层实测
   参数（旧公式漏了交叉注意力的 ``norm_q/norm_k`` ⇒ 每层少 ``2·dim`` ✓✗ 正靠这条抓）。
⑪ **装配表接入**：契约 ``(5, batch, wan21)``、``in_channels=16``（**不是** 36）、``vae_scale=8``、
   ``configSources`` 如实写「I2V 拼接未接 / 26.62 GiB 装不下」、判族仍是 ``wan``、
   ``buildable_keys`` 收录而 ``DETECT_ONLY`` 不含、表外几何不给 heads ⇒ ``BuildError`` ✓。

跑法：``.venv/Scripts/python.exe tests/engine_dit_wan22_test.py`` ✓
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:  # 本机控制台默认 cp936 ⇒ 不加兜底，首行 ✓ 就把自己崩掉了（0 条判据都跑不到）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

import torch  # noqa: E402

from app.services.engine import dit_wan22 as wan  # noqa: E402

DEV = Path(r"D:\Comfy-Desktop\ComfyUI-Shared\models\diffusion_models"
           r"\wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors")

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(f"{'✓' if ok else '✗'} {name}{('  ← ' + detail) if detail else ''}")


def tiny_shape() -> wan.WanShape:
    """小配置：``dim=48/heads=2 ⇒ head_dim 24 ⇒ 三轴 (8,8,8)`` ✓（每轴都得偶数 ✓）。"""
    return wan.WanShape(dim=48, ffn_dim=96, in_dim=36, out_dim=16, text_dim=32,
                        freq_dim=16, heads=2, layers=2, patch=(1, 2, 2))


def real_shape() -> wan.WanShape:
    """盘上 14B 的几何（不建模块 ⇒ 便宜 ✓）。"""
    return wan.WanShape(dim=5120, ffn_dim=13824, in_dim=36, out_dim=16, text_dim=4096,
                        freq_dim=256, heads=40, layers=40)


def filled(shape: wan.WanShape, *, seed: int = 20261101) -> torch.nn.Module:
    """fp64 随机权重（``modulation`` 默认全零 ⇒ 随机填，否则门控那几路测成恒等 ✗）。"""
    torch.manual_seed(seed)
    model = wan.build(shape, dtype=torch.float64).eval()
    with torch.no_grad():
        for _, param in model.named_parameters():
            param.copy_(torch.randn_like(param) * 0.15)
    return model


# --------------------------------------------------------------------------- 独立参照


def _ln(x, weight, eps, bias=None):
    """LayerNorm（``var`` 用 biased ✓，与 ``torch.nn.LayerNorm`` 一致）。

    ⚠️ ``bias`` 不能省：``norm3`` 是 ``elementwise_affine=True`` 的 **带权带偏**版本
    （盘上有 ``norm3.weight`` **和** ``norm3.bias``）。参照里只乘权重、漏掉 bias 时
    误差约 1.9e-3 —— 不报错，看着像"RoPE 没对齐"，实测差的全是这一项 ✓✗。
    """
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    y = (x - mu) * torch.rsqrt(var + eps)
    if weight is not None:
        y = y * weight
    return y if bias is None else y + bias


def _rms(x, weight, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def _linear(x, w, b):
    return torch.einsum("...i,oi->...o", x, w) + b


def _silu(x):
    return x * torch.sigmoid(x)


def _gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x.pow(3))))


def _sinusoidal(dim: int, pos: torch.Tensor) -> torch.Tensor:
    """``half = dim//2``，底数指数是 ``i/half``（**不是** ``i/dim`` ✗）。"""
    half = dim // 2
    i = torch.arange(half, dtype=torch.float64)
    omega = 10000.0 ** (-i / half)
    angle = torch.outer(pos.to(torch.float64), omega)
    return torch.cat([angle.cos(), angle.sin()], dim=1)


def _patchify(x, patch):
    b, c, t, h, w = x.shape
    pt, ph, pw = patch
    x = x.reshape(b, c, t // pt, pt, h // ph, ph, w // pw, pw)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)                    # (b, T', H', W', c, pt, ph, pw)
    return x.reshape(b, -1, c * pt * ph * pw)


def _unpatchify(x, grid, patch, out_dim):
    """⚠️ 通道在**最内层**（``(pt,ph,pw,out_dim)`` ✓）—— 与 Conv3d 展平顺序 ``(c,p,q,r)``
    **正好相反** ✓✗：写反了形状全对、图是噪的。"""
    t, h, w = grid
    pt, ph, pw = patch
    x = x.reshape(x.shape[0], t, h, w, pt, ph, pw, out_dim)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)                    # (b, c, t, pt, h, ph, w, pw)
    return x.reshape(x.shape[0], out_dim, t * pt, h * ph, w * pw)


def _attend(q, k, v):
    """显式 softmax（不用 ``scaled_dot_product_attention`` ✓ 换写法才有对拍意义）。"""
    heads, width = q.shape[-2], q.shape[-1]
    scores = torch.einsum("bshd,bthd->bhst", q, k) * (width ** -0.5)
    scores = scores - scores.amax(-1, keepdim=True)
    probs = scores.softmax(-1)
    out = torch.einsum("bhst,bthd->bshd", probs, v)
    return out.reshape(out.shape[0], out.shape[1], heads * width)


def _rope(x, grid, axes, theta):
    """逐轴直接由角度构 cos·sin（与实现的频率表写法不同 ✓），只作用于传入张量。

    频率指数 = ``linspace(0, (d-2)/d, d/2)`` ✓（``EmbedND`` 的约定 ⇒ 最高频是
    ``theta^(-(d-2)/d)`` 而**不是** ``theta^-1`` ✗；写成 ``i/half`` 会整条 RoPE 偏掉，
    症状就是"对拍差 1e-3、不报错" ✓✗）。
    """
    t_len, h_len, w_len = grid
    out = x.clone()
    start = 0
    for axis, (width, length) in enumerate(zip(axes, (t_len, h_len, w_len))):
        half = width // 2
        exponent = torch.linspace(0.0, (width - 2) / width, half, dtype=torch.float64)
        omega = theta ** (-exponent)
        angle = torch.arange(length, dtype=torch.float64)[:, None] * omega[None, :]
        view = [1, 1, 1, half]
        view[axis] = length
        angle = angle.reshape(view).expand(t_len, h_len, w_len, half).reshape(1, -1, 1, half)
        # 精度交接点（照实现声明：``rope_freqs`` 用 fp64 算频率后**降回 fp32**，
        # 参考这边同样先降到 fp32 再升回 fp64 —— 角度本身仍在 fp64 里算，只是
        # 承认这张表落地是 fp32。不镜像这一步，对拍恒定停在 ~6e-9、像是逻辑没对齐）。
        cos = angle.cos().to(torch.float32).to(torch.float64)
        sin = angle.sin().to(torch.float32).to(torch.float64)
        # ⚠️ 必须 ``clone()``：切片是视图，第一行赋值后 ``even/odd`` 已被就地改写，
        # 第二行再读就是脏值（跨轴就差 0.4546 vs 0.8415，看着像"角度算错" ✓✗）。
        even = out[..., start:start + width:2].clone()
        odd = out[..., start + 1:start + width:2].clone()
        out[..., start:start + width:2] = cos * even - sin * odd
        out[..., start + 1:start + width:2] = sin * even + cos * odd
        start += width
    return out


def _ref_forward(params, shape, latents, sigma, context, *, zero_gate=False, plus_one=True):
    """参考前向：逐句对着 ``comfy/ldm/wan/model.py`` 的 ``WanModel._forward`` 写 ✓。

    ``zero_gate`` ⇒ 六路调制全 0（测「门控有 1+ 那项」✓）；``plus_one=False`` ⇒ 故意写成 ``e``
    （证明④那条判据**能分辨** ✓）。
    """
    p = {name: value.to(torch.float64) for name, value in params.items()}
    eps = shape.eps
    b, _, t, h, w = latents.shape
    pt, ph, pw = shape.patch
    grid = (t // pt, h // ph, w // pw)
    heads, head_dim = shape.heads, shape.head_dim

    tokens = _patchify(latents, shape.patch)
    x = _linear(tokens, p["patch_embedding.weight"].reshape(shape.dim, -1),
                p["patch_embedding.bias"])
    text = _gelu_tanh(_linear(context, p["text_embedding.0.weight"],
                              p["text_embedding.0.bias"]))
    text = _linear(text, p["text_embedding.2.weight"], p["text_embedding.2.bias"])

    tv = torch.as_tensor(sigma, dtype=torch.float32).reshape(-1) * wan.TIMESTEP_SCALE
    if tv.numel() == 1:
        tv = tv.expand(b)
    # 精度交接点：实现收裸 σ 后强制 f32 走正弦、再 cast 到 latent 精度（``.to(position.dtype)``
    # ⇒ f32）⇒ 参考这边同样过一趟 f32（角度仍在 fp64 里算，只承认落地是 f32）。
    sin_emb = _sinusoidal(shape.freq_dim, tv).to(torch.float32).to(torch.float64)
    vec = _silu(_linear(sin_emb, p["time_embedding.0.weight"], p["time_embedding.0.bias"]))
    vec = _linear(vec, p["time_embedding.2.weight"], p["time_embedding.2.bias"])
    proj = _linear(_silu(vec), p["time_projection.1.weight"], p["time_projection.1.bias"])
    e_all = proj.reshape(b, 6, shape.dim)

    def self_attn(y, pre):
        q = _rms(_linear(y, p[pre + "self_attn.q.weight"], p[pre + "self_attn.q.bias"]),
                 p[pre + "self_attn.norm_q.weight"], eps).reshape(b, -1, heads, head_dim)
        k = _rms(_linear(y, p[pre + "self_attn.k.weight"], p[pre + "self_attn.k.bias"]),
                 p[pre + "self_attn.norm_k.weight"], eps).reshape(b, -1, heads, head_dim)
        v = _linear(y, p[pre + "self_attn.v.weight"],
                    p[pre + "self_attn.v.bias"]).reshape(b, -1, heads, head_dim)
        out = _attend(_rope(q, grid, shape.axes_dim, shape.theta),
                      _rope(k, grid, shape.axes_dim, shape.theta), v)   # ⚠️ v **不**旋转 ✓
        return _linear(out, p[pre + "self_attn.o.weight"], p[pre + "self_attn.o.bias"])

    def cross_attn(y, pre):
        q = _rms(_linear(y, p[pre + "cross_attn.q.weight"], p[pre + "cross_attn.q.bias"]),
                 p[pre + "cross_attn.norm_q.weight"], eps).reshape(b, -1, heads, head_dim)
        k = _rms(_linear(text, p[pre + "cross_attn.k.weight"], p[pre + "cross_attn.k.bias"]),
                 p[pre + "cross_attn.norm_k.weight"], eps).reshape(b, -1, heads, head_dim)
        v = _linear(text, p[pre + "cross_attn.v.weight"],
                    p[pre + "cross_attn.v.bias"]).reshape(b, -1, heads, head_dim)
        out = _attend(q, k, v)
        return _linear(out, p[pre + "cross_attn.o.weight"], p[pre + "cross_attn.o.bias"])

    for index in range(shape.layers):
        pre = f"blocks.{index}."
        mod = p[pre + "modulation"].reshape(6, shape.dim)
        slots = [mod[i] + e_all[:, i, :] for i in range(6)]
        if zero_gate:
            # 只清 ``modulation`` 这个**参数**，时间那一路 ``e`` 照留 —— 与实现里把
            # ``modulation`` 参数置零是同一件事（把整支 slot 清零会顺带吞掉 e，
            # 语义就变成"连时间条件也不要了"，参照与实现对不上 ✓✗）。
            slots = [e_all[:, i, :] for i in range(6)]
        # 自注意力：addcmul(e0, norm1(x), 1 + e1) ⇒ x + y·e2
        y = _ln(x, None, eps) * ((1 + slots[1]) if plus_one else slots[1])[:, None, :]
        x = x + self_attn(y + slots[0][:, None, :], pre) * slots[2][:, None, :]
        # 交叉注意力：**不带门控** ✓，用带参的 norm3 ✓
        x = x + cross_attn(_ln(x, p[pre + "norm3.weight"], eps, p[pre + "norm3.bias"]), pre)
        # FFN：addcmul(e3, norm2(x), 1 + e4) ⇒ x + y·e5
        y = _ln(x, None, eps) * ((1 + slots[4]) if plus_one else slots[4])[:, None, :]
        y = _gelu_tanh(_linear(y + slots[3][:, None, :], p[pre + "ffn.0.weight"],
                               p[pre + "ffn.0.bias"]))
        x = x + _linear(y, p[pre + "ffn.2.weight"], p[pre + "ffn.2.bias"]) * slots[5][:, None, :]

    head_mod = p["head.modulation"].reshape(2, shape.dim)
    y = _ln(x, None, eps) * (1 + head_mod[1] + vec)[:, None, :]
    out = _linear(y + (head_mod[0] + vec)[:, None, :], p["head.head.weight"],
                  p["head.head.bias"])
    return _unpatchify(out, grid, shape.patch, shape.out_dim)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / max(float(b.abs().max()), 1e-30))


def _emit(fn) -> None:
    try:
        fn()
    except Exception:  # noqa: BLE001
        check(f"{fn.__name__} 未跑完 ✗", False, traceback.format_exc().splitlines()[-1])


def case_real_header() -> None:
    """① 真权重逐键对账（只读**头**，不加载权重 ⇒ 秒级、不吃内存 ✓）。"""
    import math

    from app.services.engine import safetensors

    if not DEV.exists():
        check("① 真权重逐键对账", True, f"跳过（盘上没有 {DEV.name}）")
        return
    _, raw = safetensors.read_header(DEV)
    raw.pop("__metadata__", None)
    shapes = {name: tuple(entry["shape"]) for name, entry in raw.items()}
    dtypes = {name: entry["dtype"] for name, entry in raw.items()}

    shape = wan.infer_shape(shapes)
    keys = wan.expected_keys(shape)
    geometry = (shape.layers, shape.dim, shape.ffn_dim, shape.heads, shape.head_dim,
                shape.axes_dim, shape.in_dim, shape.out_dim, shape.text_dim,
                shape.freq_dim, shape.patch)
    check("① 几何与盘上一致（40 块 / 5120 / 13824 / 40 头 / 128 · 三轴 44·42·42 / 36 / 16 / 4096 / 256 / (1,2,2)）",
          geometry == (40, 5120, 13824, 40, 128, (44, 42, 42), 36, 16, 4096, 256, (1, 2, 2)),
          str(geometry))

    check("①′ 头里 1908 条 = 真权重 1095 + fp8 伴生 813",
          (len(keys), len(shapes)) == (1095, 1908), f"{len(keys)} / {len(shapes)}")
    missing = keys - set(shapes)
    check("①″ 缺键 0（反过来：真权重一条不缺）", not missing, f"缺 {sorted(missing)[:3]}")

    extra = sorted(set(shapes) - keys)
    companions = {name for name in extra
                  if name.endswith((".scale_weight", ".scale_input"))}
    odd = [name for name in extra if name not in companions]
    check("①‴ 813 条余项**全部**是 fp8 伴生（406 scale_input + 406 scale_weight + 1 哨兵）",
          len(extra) == 813 and len(companions) == 812 and odd == ["scaled_fp8"],
          f"余 {len(extra)} / 伴生 {len(companions)} / 异类 {odd[:3]}")

    quant = sum(1 for name in keys if dtypes[name] == "F8_E4M3")
    check("①⁗ 量化权重 406 张 fp8 + 689 张 F16 ⇒ 省盘不省卡（反量化到 bf16 才是驻留量）",
          quant == 406 and sum(1 for name in keys if dtypes[name] == "F16") == 689,
          f"F8 {quant} / F16 {689}")

    counted = sum(math.prod(shapes[name]) for name in keys)
    check("①⁵ 参数量 = 头里逐张量 numel 之和 = 14 288 901 184",
          wan.param_count(shape) == counted == 14_288_901_184, f"{wan.param_count(shape):,}")
    gib = wan.param_count(shape) * 2 / 1024 ** 3
    check(f"①⁶ 结论：bf16 常驻 ≈{gib:.2f} GiB ⇒ 22.49 GiB 的卡**装不下**（本机真跑不了）",
          gib > 22.49, f"{gib:.2f} GiB")


def case_shape_roundtrip() -> None:
    """② 形状 → 模块 → 装载 闭环（这正是装载器的三步 ✓）。"""
    shape = tiny_shape()
    model = wan.build(shape, dtype=torch.float64)
    state = {name: tuple(t.size()) for name, t in model.state_dict().items()}
    inferred = wan.infer_shape(state, heads=shape.heads)
    check("② state_dict 反推形状 == 原形状（patch 从 Conv3d 权重读 ✓、layers 从键名数 ✓）",
          inferred == shape, "" if inferred == shape else f"{inferred} ≠ {shape}")
    keys_ok = wan.expected_keys(inferred) == set(state)
    rebuilt = wan.build(inferred, dtype=torch.float64)
    report = rebuilt.load_state_dict(model.state_dict(), strict=True)
    check("②′ expected_keys 与 state_dict 逐键相等，strict=True 装载无缺无余",
          keys_ok and not report.missing_keys and not report.unexpected_keys,
          f"缺 {list(report.missing_keys)[:2]} 余 {list(report.unexpected_keys)[:2]}")


def case_parity() -> None:
    """③ 前向 vs 独立参照（fp64 对 fp64，误差 ≈ 1e-13）。"""
    shape = tiny_shape()
    model = filled(shape)
    torch.manual_seed(11)
    latents = torch.randn(2, 36, 3, 4, 4, dtype=torch.float64)
    context = torch.randn(2, 5, shape.text_dim, dtype=torch.float64)
    sigma = torch.tensor([0.75, 0.5])
    out = model(latents, sigma, context=context)
    ref = _ref_forward(model.state_dict(), shape, latents, sigma, context)
    err = _rel(out, ref)
    check("③ fp64 前向与独立参照一致（相对误差 < 1e-9）", err < 1e-9, f"误差 {err:.2e}")
    check("③′ 输出形状 = (b, 16, t, h, w)（patch (1,2,2) ⇒ 空间各缩一半）",
          tuple(out.shape) == (2, 16, 3, 4, 4), str(tuple(out.shape)))

    single = latents[:1, :, :1]
    out1 = model(single, 0.5, context=context[:1])
    ref1 = _ref_forward(model.state_dict(), shape, single, 0.5, context[:1])
    check("③″ t=1 单帧 × sigma 标量 的路径也一致", _rel(out1, ref1) < 1e-9,
          f"误差 {_rel(out1, ref1):.2e}")


def case_gate() -> None:
    """④ 门控里的 ``1 +``：置零 modulation ⇒ 退化成无门控，而不是整支乘 0。"""
    shape = tiny_shape()
    model = filled(shape)
    with torch.no_grad():
        for block in model.blocks:
            block.modulation.zero_()
        model.head.modulation.zero_()
    torch.manual_seed(13)
    latents = torch.randn(1, 36, 2, 4, 4, dtype=torch.float64)
    context = torch.randn(1, 4, shape.text_dim, dtype=torch.float64)
    out = model(latents, 0.6, context=context)
    gapless = _ref_forward(model.state_dict(), shape, latents, 0.6, context, zero_gate=True)
    check("④ modulation 全零 ⇒ 等于「无门控」参照（addcmul 的 1 那项在）",
          _rel(out, gapless) < 1e-9, f"误差 {_rel(out, gapless):.2e}")

    without = _ref_forward(model.state_dict(), shape, latents, 0.6, context,
                           zero_gate=True, plus_one=False)
    check("④′ 反向：漏了 ``1 +``（写成 ``e``）的那版参照**对不上** ⇒ 判据有分辨力",
          _rel(out, without) > 1e-3, f"误差 {_rel(out, without):.2e}")

    live = filled(shape)
    out_live = live(latents, 0.6, context=context)
    check("④″ modulation 非零时输出确实变（六路真参与前向，不是摆设）",
          _rel(out_live, out) > 1e-3, f"差 {_rel(out_live, out):.2e}")


def case_rope() -> None:
    """⑤ 3D RoPE：位置号不居中 / 三轴宽度 / 只转 q·k / 轴宽必须偶数。"""
    shape = tiny_shape()
    check("⑤ 三轴宽度：head_dim 128 ⇒ (44,42,42)、小配置 24 ⇒ (8,8,8)",
          real_shape().axes_dim == (44, 42, 42) and shape.axes_dim == (8, 8, 8),
          f"{real_shape().axes_dim} / {shape.axes_dim}")

    single = wan.rope_freqs(shape, 1, 1, 1)
    cos0, sin0 = single[..., 0, 0], single[..., 1, 0]
    check("⑤′ 1×1×1 网格的旋转表恰是恒等 ⇒ 位置号**从 0 起、不居中**（居中会是 -1/2 位）",
          bool(torch.allclose(cos0, torch.ones_like(cos0))
               and torch.allclose(sin0, torch.zeros_like(sin0))))
    check("⑤″ 频率表形状 [1, L, 1, head_dim/2, 2, 2]（第 3 维给头维广播 ✓）",
          tuple(single.shape) == (1, 1, 1, shape.head_dim // 2, 2, 2), str(tuple(single.shape)))

    stepped = wan.rope_freqs(shape, 2, 1, 1)
    quarter = shape.axes_dim[0] // 2                 # 三轴每段各 head_dim/2 ÷ 3 宽
    t_seg = stepped[0, 1, 0, :quarter, 0, 0]
    hw_seg = stepped[0, 1, 0, quarter:, 0, 0]
    check("⑤‴ t 轴长度 2 ⇒ **只有 t 那一段**角度变、h/w 两段仍是 1（三轴宽度分配对）",
          not torch.allclose(t_seg, torch.ones_like(t_seg))
          and torch.allclose(hw_seg, torch.ones_like(hw_seg)),
          f"t 段 {t_seg.tolist()[:2]} / 其余 {hw_seg.tolist()[:2]}")

    torch.manual_seed(17)
    x = torch.randn(1, 6, 3, shape.head_dim, dtype=torch.float64)
    rotated = wan.apply_rope(x, wan.rope_freqs(shape, 3, 2, 1))

    def _norms(t):
        return t.reshape(*t.shape[:-1], -1, 2).pow(2).sum(-1).sqrt()

    check("⑤⁗ 旋转保每对 2 分量的范数（正交块 ✓），且确实改变了张量",
          torch.allclose(_norms(rotated), _norms(x), atol=1e-12)
          and not torch.allclose(rotated, x))

    model = wan.build(shape, dtype=torch.float64)
    block = model.blocks[0]
    check("⑤⁵ 结构：自注意力走 RoPE、交叉注意力**不**走（v 也不转）",
          block.self_attn.rope is True and block.cross_attn.rope is False)

    try:
        wan.WanShape(dim=4, ffn_dim=8, in_dim=36, out_dim=16, text_dim=8, freq_dim=8,
                     heads=2, layers=1)
        raised = ""
    except Exception as exc:  # noqa: BLE001
        raised = str(exc)
    check("⑤⁶ head_dim=2 ⇒ 三轴里有 0 宽轴 ⇒ 当场抛（不静默建出坏 RoPE）",
          "轴" in raised or "axes" in raised, raised[:60])


def _real_key_shapes(layers: int = 40) -> dict[str, tuple[int, ...]]:
    """14B 的**最小键集**（只喂 ``infer_shape`` 要读的那几格 ⇒ 不建模型 ✓）。"""
    shapes: dict[str, tuple[int, ...]] = {
        "patch_embedding.weight": (5120, 36, 1, 2, 2),
        "head.head.weight": (64, 5120),
        "text_embedding.0.weight": (5120, 4096),
        "time_embedding.0.weight": (5120, 256),
        "blocks.0.norm3.weight": (5120,),
        "blocks.0.self_attn.norm_q.weight": (5120,),
    }
    for index in range(layers):
        shapes[f"blocks.{index}.ffn.0.weight"] = (13824, 5120)
    return shapes


def case_heads() -> None:
    """⑥ heads 权重里读不到 ⇒ 只许查表，不许猜比例。"""
    check("⑥ 查表只有一条真几何 (5120,13824,40) ⇒ 40",
          wan.HEADS_BY_GEOMETRY == {(5120, 13824, 40): 40})
    shape = tiny_shape()
    model = wan.build(shape, dtype=torch.float64)
    state = {name: tuple(t.size()) for name, t in model.state_dict().items()}
    try:
        wan.infer_shape(state)
        raised = ""
    except Exception as exc:  # noqa: BLE001
        raised = str(exc)
    check("⑥′ 表外几何不给 heads ⇒ 具名抛（文案点名 heads）", "heads" in raised, raised[:70])
    check("⑥″ 显式给 heads ⇒ 正常推出",
          wan.infer_shape(state, heads=shape.heads).heads == shape.heads)
    big = wan.infer_shape(_real_key_shapes())
    check("⑥‴ 真几何不给 heads 也能推（走 (dim,ffn,layers) 查表 ⇒ 40，不是 dim//128 猜）",
          big.heads == 40 and big.head_dim == 128, f"heads {big.heads} / head_dim {big.head_dim}")


def case_strictness() -> None:
    """⑦ 七种错形状都必须**具名**抛，不许静默补边/猜。"""
    shape = tiny_shape()
    model = wan.build(shape, dtype=torch.float64)
    good = torch.zeros(1, 36, 1, 4, 4, dtype=torch.float64)
    ctx = torch.zeros(1, 3, shape.text_dim, dtype=torch.float64)

    def _err(**kwargs) -> str:
        try:
            kwargs.setdefault("latents", good)
            kwargs.setdefault("sigma", torch.tensor([0.5]))
            kwargs.setdefault("context", ctx)
            model(**kwargs)
        except Exception as exc:  # noqa: BLE001
            return str(exc)
        return ""

    lat4 = _err(latents=torch.zeros(1, 16, 4, 4, dtype=torch.float64))
    chan = _err(latents=torch.zeros(1, 16, 1, 4, 4, dtype=torch.float64))
    no_ctx = _err(context=None)
    bad_ctx = _err(context=torch.zeros(1, 3, shape.text_dim - 1, dtype=torch.float64))
    pad = _err(latents=torch.zeros(1, 36, 1, 5, 4, dtype=torch.float64))
    sig = _err(sigma=torch.tensor([0.5, 0.4]))
    check("⑦ latent 4 维 ⇒ 抛（4 维是出图族的口径）", "5 维" in lat4, lat4[:60])
    check("⑦′ 通道 16 ⇒ 抛（只喂噪声、漏了条件图与掩码）", "通道" in chan, chan[:60])
    check("⑦″ context 缺失 / 宽度错 ⇒ 抛（不落成 text_embedding 的 matmul 错）",
          "context" in no_ctx and "context" in bad_ctx, (no_ctx or bad_ctx)[:60])
    check("⑦‴ h=5 不是 patch(2) 的整数倍 ⇒ 抛（**不补边**：补边等于悄悄改构图）",
          "整数倍" in pad, pad[:70])
    check("⑦⁗ sigma 个数 ≠ 批 ⇒ 抛", "sigma" in sig, sig[:60])

    bf = wan.build(shape, dtype=torch.bfloat16)
    ctx_bf = torch.zeros(1, 3, shape.text_dim, dtype=torch.bfloat16)
    out = bf(torch.zeros(1, 36, 1, 4, 4, dtype=torch.bfloat16), torch.tensor([0.5]),
             context=ctx_bf)
    check("⑦⁵ bf16 进 bf16 出（RoPE 的 fp32 频率不会把 q/k 升精度 ⇒ SDPA 不报 dtype 不合）",
          out.dtype == torch.bfloat16, str(out.dtype))


def case_timestep() -> None:
    """⑧ σ ⇒ 时刻的 ×1000 必须在骨架里（参考实现把它放在采样器里）。"""
    shape = tiny_shape()
    model = filled(shape)
    seen: dict[str, torch.Tensor] = {}

    def _spy(_mod, args, _out):
        # ⚠️ 钩子**不能有返回值**：return 一个张量等于把模块输出替换掉（``unflatten`` 当场炸）
        seen["v"] = args[0]

    # ⚠️ 钓 ``time_embedding`` 的**输入**：``time_projection`` 的输入是
    # ``time_embedding`` 的输出（宽 ``dim``），挂在那儿只能比到 dim 宽的东西，
    # 与 freq_dim 的正弦做减法当场 ``size a(48) vs b(16)`` ✓✗。
    handle = model.time_embedding.register_forward_hook(_spy)
    try:
        model(torch.zeros(1, 36, 1, 4, 4, dtype=torch.float64), torch.tensor([0.75]),
              context=torch.zeros(1, 2, shape.text_dim, dtype=torch.float64))
    finally:
        handle.remove()
    got = seen["v"].reshape(1, -1).to(torch.float64)
    scaled = wan.sinusoidal_embedding_1d(shape.freq_dim, torch.tensor([0.75 * 1000.0]))
    raw = wan.sinusoidal_embedding_1d(shape.freq_dim, torch.tensor([0.75]))
    check("⑧ TIMESTEP_SCALE = 1000（ModelSamplingDiscreteFlow.multiplier 的默认值）",
          wan.TIMESTEP_SCALE == 1000.0)
    check("⑧′ time_embedding 吃到的正弦 == sin(freq_dim, 1000·σ)",
          float((got - scaled).abs().max()) < 1e-4, f"差 {float((got - scaled).abs().max()):.2e}")
    check("⑧″ 反向：**不**等于 sin(freq_dim, σ) 那版（漏乘 1000 会被这条抓住）",
          float((got - raw).abs().max()) > 1e-2, f"差 {float((got - raw).abs().max()):.2e}")


def case_split_i2v() -> None:
    """⑨ 36 通道布局：切片宽度、拼回一致、错通道抛。"""
    shape = tiny_shape()
    torch.manual_seed(19)
    x = torch.randn(1, 36, 2, 4, 4, dtype=torch.float64)
    parts = wan.split_i2v(shape, x)
    widths = {name: tuple(t.shape[1:2]) for name, t in parts.items()}
    check("⑨ 36 = 噪声 16 + 条件图 16 + 掩码 4（掩码 4 是因为时间下采样并进通道维）",
          widths == {"noise": (16,), "image": (16,), "mask": (4,)}, str(widths))
    rejoined = torch.cat([parts[name] for name, _ in wan.I2V_LAYOUT], dim=1)
    check("⑨′ 按声明顺序拼回逐字节相等（切分没串位）", rejoined.equal(x))
    try:
        wan.split_i2v(shape, torch.zeros(1, 32, 2, 4, 4))
        raised = ""
    except Exception as exc:  # noqa: BLE001
        raised = str(exc)
    check("⑨″ 通道不是 36 ⇒ 抛（顺手把 Wan2.1 的 32 也拦下）", "通道" in raised, raised[:60])


def case_param_count() -> None:
    """⑩ param_count 公式与实测逐层对得上（旧版漏了交叉注意力的 norm_q/norm_k）。"""
    import dataclasses

    shape = tiny_shape()
    model = wan.build(shape, dtype=torch.float64)
    real = sum(p.numel() for p in model.parameters())
    check("⑩ param_count == 建出来的参数量", wan.param_count(shape) == real,
          f"{wan.param_count(shape):,} vs {real:,}")
    block = sum(p.numel() for name, p in model.named_parameters()
                if name.startswith("blocks.0."))
    base = {field.name: getattr(shape, field.name) for field in dataclasses.fields(shape)}
    one = wan.param_count(wan.WanShape(**{**base, "layers": 1}))
    two = wan.param_count(wan.WanShape(**{**base, "layers": 2}))
    check("⑩′ 每加一层的增量 == 该层实测参数（每层少算 2·dim 就会露馅）",
          two - one == block, f"增量 {two - one:,} / 实测 {block:,}")
    check("⑩″ 真几何参数量 = 14 288 901 184（与①⁵ 互核：公式没碰运气）",
          wan.param_count(real_shape()) == 14_288_901_184, f"{wan.param_count(real_shape()):,}")


def case_build_table() -> None:
    """⑪ 装配表接入：契约、in_channels、configSources 如实、判族、表外 heads。"""
    from app.services.engine import dit_build as bd
    from app.services.engine import dit_family
    from app.services.engine.safetensors import SafetensorsInfo, TensorEntry

    shape = tiny_shape()
    model = wan.build(shape, dtype=torch.float64)
    pairs = {name: tuple(t.size()) for name, t in model.state_dict().items()}
    info = SafetensorsInfo(path="<合成>", tensors={
        name: TensorEntry(name=name, dtype="BF16", shape=list(shp), begin=0, end=0)
        for name, shp in pairs.items()})

    contract = bd.contract_of("wan")
    check("⑪ 契约 = (潜变量 5 维, σ 逐批, 口径 wan21)",
          (contract.latent_dims, contract.sigma_form, contract.latent_format)
          == (5, "batch", "wan21"),
          f"{contract.latent_dims}/{contract.sigma_form}/{contract.latent_format}")

    built = bd.build("wan", info, {"heads": shape.heads}, dtype=torch.float32)
    module, config = built if built is not None else (None, {})
    check("⑪′ 建得出，且 in_channels=16（潜宽，**不是**主干的 36）、vae_scale=8、text_dim=32",
          module is not None and (config.get("in_channels"), config.get("vae_scale"),
                                  config.get("text_dim")) == (16, 8, 32),
          str({k: config.get(k) for k in ("in_channels", "vae_scale", "text_dim")}))

    sources = config.get("configSources", {})
    check("⑪″ configSources 如实写「I2V 拼接未接」「28.6 GiB 装不下」",
          "未接" in sources.get("i2vConcat", "") and "装不下" in sources.get("residency", ""),
          f"{sources.get('i2vConcat', '')[:20]} | {sources.get('residency', '')[:20]}")

    try:
        bd.build("wan", info, {}, dtype=torch.float32)
        raised = ""
    except bd.BuildError as exc:
        raised = str(exc)
    check("⑪‴ 表外几何不给 heads ⇒ BuildError（把 ShapeError 包成人话，不猜 dim//128）",
          "heads" in raised, raised[:80])

    check("⑪⁗ 判族仍判成 wan、且已在 buildable_keys 里（不再走「只判族不建」的老路）",
          dit_family.detect_from_keys(list(pairs)) == "wan"
          and "wan" in bd.buildable_keys(),
          str(dit_family.detect_from_keys(list(pairs))))


def main() -> int:
    print("== Wan2.2 I2V DiT 骨架自检 ==")
    for case in (case_real_header, case_shape_roundtrip, case_parity, case_gate,
                 case_rope, case_heads, case_strictness, case_timestep, case_split_i2v,
                 case_param_count, case_build_table):
        _emit(case)
    failed = [name for name, ok, _ in results if not ok]
    print(f"SUMMARY: {len(results) - len(failed)}/{len(results)} passed"
          + (f" ✗ 失败：{failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
