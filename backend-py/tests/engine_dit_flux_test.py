"""Flux.1 DiT 骨架自检（``dit_flux.py``）✓ —— 判族 ✅ / 建壳 ✅ / 前向 ✅ / 精度 ✅。

九条判据（每条都能**单独**失败 ✓；默认 32 项断言，开 ⑧ 后 37 项）：

① **真权重逐键对账**：``flux1-dev.safetensors`` 780 键 == ``expected_keys`` ✓，
   层数 19/38、heads 24、head_dim 128、``img_in`` 入宽 64、``single.linear1`` 21504 ✓。
   盘上没有这个文件 ⇒ **跳过**（不算失败 ✓，但摘要里会写明「跳过」✗ 不假装跑过 ✓）。
② **形状 → 模块 → 装载**闭环：用小配置自身 ``state_dict`` 反推形状 ✓，再 ``infer_shape`` →
   ``build`` → ``load_state_dict(strict=True)`` ✓（这正是装载器的三步 ✓）。
③ **前向与独立参照双精度一致** ✓：参照实现**刻意换写法**（逐 token 显式 softmax /
   直接由角度构 cos·sin / 手写 gelu-tanh 公式 ✓）⇒ 相对误差 < 1e-7（实测 floor ≈ 2e-9，
   全部来自实现里**刻意**的两处 fp32：RoPE 频率与时间角度 ✓ —— 理由见 ``case_parity``）。
④ **拼接顺序 ``(txt, img)``** ✓：``_attn_mask`` 的非零列必须**恰好**落在前 ``n_txt`` 列 ✓；
   并且「挡掉文本 pad 位」必须真的改变输出 ✓（否则掩码接错了位置也不报错 ✓✗）。
⑤ **三路 vec**：guidance 变 → 输出变 ✓；pooled 向量变 → 输出变 ✓；两者都不给 → 走默认 ✓。
⑥ **口径**：``1000·σ``（σ=0 与 σ=1 的时刻嵌入不同且**不互换** ✓）、位置号**不居中**
   （1×1 latent 的 h/w 号都是 0 ⇒ 旋转是恒等 ✓；2×1 的两个 token 号不同 ✓）、
   ``FluxShape.t_scale`` **是活字段**（改 1000→1 输出必须变 ✓✗ —— 前向漏传 `time_factor` 时
   行为一模一样、字段变成摆设 ✗）。
⑦ **严格性**：context 宽错 / latent 通道错 / latent 尺寸非 patch 倍数 / vector 太窄 ⇒
   四条都必须**具名抛** ✓；bf16 进 bf16 出 ✓。
⑧ **真权重整份装载**（默认**跳过** ✗ —— 22 GiB 驻留，不能进 ``run_all``）：
   ``ENGINE_FLUX_REAL=1`` 开 ⇒ ``dit_build`` 真建 + 780 键真装 + 真尺寸前向 + guidance 通路 ✓。
   **① 与 ⑧ 不是一件事**：① 只比键名形状、② 只在小配置上闭环；只有真文件能证明装配表那行接对了 ✓。
⑨ **``default_axes`` 的自洽性**：任意偶数 ``head_dim`` 都得出三个**非零偶数**、和恰为 ``head_dim`` ✓，
   关键档位（128/64/32/16/8/12/10）与文档逐字一致 ✓，``head_dim=16`` 能**真的**建出 RoPE ✓。
   ⚠️ 这条是 2026-11-01 复查补的判据：原实现两轮分支判据都算出了奇数轴（16 ⇒ 2/7/7、12 ⇒ 3/6/3），
   症状是「建得成、跑不动」，而报错点落在 ``axes_dim`` 上、像是调用方传错 ✗。
⑩ **``attach_flux_conditioning`` 的回执可信**：挂载是写进模型属性的 ⇒ 第二次空调用时上一次的
   ``guidance``/``vector`` **还在生效** ✓；回执必须报「真在生效」的值，不许一律报
   ``guidance_default`` 与「零向量」（那是「以为退回 3.5、实际还在 7.0」✗）。

跑法：``.venv/Scripts/python.exe tests/engine_dit_flux_test.py`` ✓
真权重（慢、吃 22 GiB 内存）：``ENGINE_FLUX_REAL=1 .venv/Scripts/python.exe tests/engine_dit_flux_test.py`` ✓
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:  # 本机控制台默认 cp936 ⇒ 不加兜底，首行 ✓ 就把自己崩掉了（0 条判据都跑不到）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

import torch  # noqa: E402

from app.services.engine import dit_flux as flux  # noqa: E402
from app.services.engine import dit_generic as generic  # noqa: E402

DEV = Path(r"D:\Comfy-Desktop\ComfyUI-Shared\models\unet\flux1-dev.safetensors")

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(f"{'✓' if ok else '✗'} {name}{('  ← ' + detail) if detail else ''}")


def tiny_shape() -> flux.FluxShape:
    """小配置（head_dim 8 = 2+4+2 ✓ 三轴必须**偶数** ✓：一轴奇数 ⇒ rope 按 2 分组对不齐 ✗）。"""
    return flux.FluxShape(features=32, double_layers=2, single_layers=3, heads=4,
                          head_dim=8, channels=8, context_dim=12, vec_dim=6,
                          mlp_hidden=48, axes_dim=(2, 4, 2))


# ---------------------------------------------------------------- ① 真权重


def case_real_header() -> None:
    if not DEV.exists():
        check("① 真权重逐键对账", True, f"跳过（盘上没有 {DEV.name} ✗ 不算通过）")
        return
    from app.services.engine import safetensors as st  # noqa: PLC0415

    _, raw = st.read_header(str(DEV))
    shapes = {k: v["shape"] for k, v in raw.items() if k != "__metadata__"}
    shape = flux.infer_shape(shapes)
    keys = flux.expected_keys(shape)
    check("① 真权重逐键对账（780 == expected_keys）", keys == set(shapes),
          f"实得 {len(shapes)} 键 / 期望 {len(keys)}；"
          f"缺 {sorted(set(shapes) - keys)[:3]} 多 {sorted(keys - set(shapes))[:3]}")
    check("① 层数与头数", (shape.double_layers, shape.single_layers, shape.heads,
                        shape.head_dim) == (19, 38, 24, 128),
          f"{shape.double_layers}/{shape.single_layers}/{shape.heads}/{shape.head_dim}")
    check("① 通道 / 文本宽 / pooled 宽 / mlp",
          (shape.channels, shape.patch_in, shape.context_dim, shape.vec_dim,
           shape.mlp_hidden) == (16, 64, 4096, 768, 12288),
          f"{shape.channels}/{shape.patch_in}/{shape.context_dim}/{shape.vec_dim}/"
          f"{shape.mlp_hidden}")
    total = sum(int(torch.tensor(v["shape"]).prod()) if v["shape"] else 1
                for v in raw.values() if isinstance(v, dict) and "shape" in v)
    check("① 参数量 ≈ 11.9 G（22.17 GiB bf16）", 11.5e9 < total < 12.1e9,
          f"{total / 1e9:.2f} G 参数 ⇒ bf16 {total * 2 / 2 ** 30:.2f} GiB")


# ---------------------------------------------------------------- ② 形状闭环


def case_shape_roundtrip() -> None:
    shape = tiny_shape()
    model = flux.build(shape)
    shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    again = flux.infer_shape(shapes)
    check("② 形状反推与建网一致", again == shape,
          f"{again} vs {shape}")
    keys = flux.expected_keys(again)
    check("② 键集合与 state_dict 相等", keys == set(shapes),
          f"缺 {sorted(set(shapes) - keys)[:3]} 多 {sorted(keys - set(shapes))[:3]}")
    rebuilt = flux.build(again)
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    check("② 严格装载（strict=True）无缺失/多余", True)


# ---------------------------------------------------------------- ③ 独立参照


def _ref_gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def _ref_rms(x: torch.Tensor, scale: torch.Tensor, eps: float) -> torch.Tensor:
    """**qk-norm 用**：无均值、无 ``+1``（权重就是 scale 本身）✓。"""
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps) * scale


def _ref_ln(x: torch.Tensor, eps: float) -> torch.Tensor:
    """**块内 pre-norm 用**：``nn.LayerNorm(elementwise_affine=False)`` ⇒ **减均值**、除以总体标准差。

    ⚠️ 这一格最容易写错 ✓✗：Flux 的块内归一化是 LayerNorm（有均值、有 var）✓，只有
    ``query_norm`` / ``key_norm`` 才是 RMSNorm ✓。参照里若把 pre-norm 也写成 RMS（不减均值），
    两条路径的差会以「相对误差 0.35」的形状出现 ✓ —— 看着像实现坏了，其实是参照错了 ✗。
    """
    return (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + eps)


def _ref_timestep(t: torch.Tensor, dim: int) -> torch.Tensor:
    """参照版：直接由角度构 cos/sin ✓（不调仓里的 ``timestep_embedding`` ✗ —— 换写法才有对拍意义 ✓）。

    ⚠️ 角度必须**在 fp32 里算**再升 fp64 ✓✗：guidance 3.5 走 ``×1000 = 3500 rad`` ✓，
    fp32 在 3500 附近的角度分辨率约 2e-4 rad ✓ ⇒ 若参照用 fp64 算角度，与实现（fp32）
    会差到 1e-1 量级 ✓ —— 那不是实现错、而是这一族**固有的精度事实** ✓（参考实现也走 fp32 ✓）。
    """
    half = dim // 2
    idx = torch.arange(half, dtype=torch.float32)
    omega = torch.exp(-math.log(10000.0) * idx / half)
    ang = (t.float() * 1000.0).reshape(-1, 1) * omega.reshape(1, -1)
    return torch.cat((torch.cos(ang), torch.sin(ang)), dim=-1).double()


def _ref_rope(x: torch.Tensor, ids: torch.Tensor, axes: tuple[int, ...],
              theta: float) -> torch.Tensor:
    """参照版：按轴逐段构角度 ⇒ 标准旋转 ``(x0·c − x1·s, x0·s + x1·c)`` ✓。"""
    parts = []
    for axis, dim in enumerate(axes):
        pos = ids[..., axis].double()
        idx = torch.arange(0, dim, 2, dtype=torch.float64)
        ang = pos.unsqueeze(-1) * torch.exp(-math.log(theta) * idx / dim)
        c, s = torch.cos(ang), torch.sin(ang)
        seg = x[..., sum(axes[:axis]):sum(axes[:axis + 1])].double()
        pair = seg.reshape(*seg.shape[:-1], -1, 2)
        rot = torch.stack((pair[..., 0] * c - pair[..., 1] * s,
                           pair[..., 0] * s + pair[..., 1] * c), dim=-1)
        parts.append(rot.reshape(*seg.shape))
    return torch.cat(parts, dim=-1)


def _ref_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   ids: torch.Tensor, axes: tuple[int, ...], theta: float,
                   keep: torch.Tensor | None) -> torch.Tensor:
    """参照版：显式 ``softmax(qkᵀ/√d)`` ✓（不走 SDPA ✓），掩码用**加法 -inf** ✓。"""
    q = _ref_rope(q, ids, axes, theta)
    k = _ref_rope(k, ids, axes, theta)
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if keep is not None:
        bias = torch.zeros(scores.shape[-2:], dtype=torch.float64)
        bias[:, :keep.shape[1]] = torch.where(keep[0], 0.0, float("-inf")).double()
        scores = scores + bias
    w = torch.softmax(scores, dim=-1)
    return w @ v


def _ref_forward(model: flux.FluxDiT, latents: torch.Tensor, sigma: torch.Tensor,
                 context: torch.Tensor, vector: torch.Tensor | None,
                 guidance: float | None) -> torch.Tensor:
    """独立参照（fp64、逐块手写）✓ —— 与 ``FluxDiT.forward`` **不是**同一份代码路径 ✓。"""
    sd = {k: v.double() for k, v in model.state_dict().items()}
    s = model.shape
    b, c, h, w = latents.shape
    p = s.patch
    hh, ww = h // p, w // p
    x = latents.double()
    # patchify：逐 patch 显式切片（不用 view/permute 那套 ✓）
    rows = []
    for i in range(hh):
        for j in range(ww):
            rows.append(x[:, :, i * p:(i + 1) * p, j * p:(j + 1) * p].reshape(b, -1))
    img = torch.stack(rows, dim=1)
    ids = torch.zeros(1, context.shape[1] + hh * ww, 3, dtype=torch.float64)
    for i in range(hh):
        for j in range(ww):
            ids[0, context.shape[1] + i * ww + j] = torch.tensor([0.0, float(i), float(j)])

    def lin(name: str, val: torch.Tensor) -> torch.Tensor:
        return val @ sd[f"{name}.weight"].T + sd[f"{name}.bias"]

    def embed(name: str, val: torch.Tensor) -> torch.Tensor:
        return lin(f"{name}.out_layer", torch.nn.functional.silu(lin(f"{name}.in_layer", val)))

    img = lin("img_in", img)
    vec = embed("time_in", _ref_timestep(sigma, s.t_freq))
    if "guidance_in.in_layer.weight" in sd:
        g = torch.full((b, 1), float(flux.GUIDANCE_DEFAULT if guidance is None else guidance),
                       dtype=torch.float64)
        vec = vec + embed("guidance_in", _ref_timestep(g.reshape(-1), s.t_freq))
    if vector is not None:
        vec = vec + embed("vector_in", vector.double()[:, :s.vec_dim])
    txt = lin("txt_in", context.double())

    heads, hd = s.heads, s.head_dim
    vec3 = vec[:, None, :]   # 调制按 token 广播 ✓（模块里在 _Modulation 内部就做这一步 ✓）

    def to_heads(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(*t.shape[:-1], heads, hd).transpose(-3, -2)

    def from_heads(t: torch.Tensor) -> torch.Tensor:
        moved = t.transpose(-3, -2)
        return moved.reshape(*moved.shape[:-2], heads * hd)

    for i in range(s.double_layers):
        pre = f"double_blocks.{i}"
        for stream, val in (("img", img), ("txt", txt)):
            mod = lin(f"{pre}.{stream}_mod.lin",
                      torch.nn.functional.silu(vec3))
            shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=-1)
            normed = _ref_ln(val, s.eps)
            normed = normed * (1 + scale1) + shift1
            qkv = lin(f"{pre}.{stream}_attn.qkv", normed)
            q, k, v = qkv.chunk(3, dim=-1)
            q, k = to_heads(q), to_heads(k)
            q = _ref_rms(q, sd[f"{pre}.{stream}_attn.norm.query_norm.scale"], s.eps)
            k = _ref_rms(k, sd[f"{pre}.{stream}_attn.norm.key_norm.scale"], s.eps)
            if stream == "img":
                q_img, k_img, v_img = q, k, v
            else:
                q_txt, k_txt, v_txt = q, k, v
        n_txt = txt.shape[1]
        q_all = torch.cat((q_txt, q_img), dim=-2)
        k_all = torch.cat((k_txt, k_img), dim=-2)
        v_all = torch.cat((to_heads(v_txt), to_heads(v_img)), dim=-2)
        out = _ref_attention(q_all, k_all, v_all, ids, s.axes_dim, s.theta, None)
        img_attn, txt_attn = out[..., n_txt:, :], out[..., :n_txt, :]
        for stream, val, at in (("img", img, img_attn), ("txt", txt, txt_attn)):
            mod = lin(f"{pre}.{stream}_mod.lin", torch.nn.functional.silu(vec3))
            shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=-1)
            val = val + gate1 * lin(f"{pre}.{stream}_attn.proj", from_heads(at))
            normed = _ref_ln(val, s.eps)
            normed = normed * (1 + scale2) + shift2
            mlp = lin(f"{pre}.{stream}_mlp.2",
                      _ref_gelu_tanh(lin(f"{pre}.{stream}_mlp.0", normed)))
            val = val + gate2 * mlp
            if stream == "img":
                img = val
            else:
                txt = val

    x = torch.cat((txt, img), dim=1)
    for i in range(s.single_layers):
        pre = f"single_blocks.{i}"
        mod = lin(f"{pre}.modulation.lin", torch.nn.functional.silu(vec3))
        shift, scale, gate = mod.chunk(3, dim=-1)
        normed = _ref_ln(x, s.eps)
        normed = normed * (1 + scale) + shift
        qkv_mlp = lin(f"{pre}.linear1", normed)
        qkv, mlp = qkv_mlp.split([3 * s.features, s.mlp_hidden], dim=-1)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k = to_heads(q), to_heads(k)
        q = _ref_rms(q, sd[f"{pre}.norm.query_norm.scale"], s.eps)
        k = _ref_rms(k, sd[f"{pre}.norm.key_norm.scale"], s.eps)
        out = _ref_attention(q, k, to_heads(v), ids, s.axes_dim, s.theta, None)
        joined = torch.cat((from_heads(out), _ref_gelu_tanh(mlp)), dim=-1)
        x = x + gate * lin(f"{pre}.linear2", joined)

    tail = x[:, txt.shape[1]:]
    mod = lin("final_layer.adaLN_modulation.1", torch.nn.functional.silu(vec3))
    shift, scale = mod.chunk(2, dim=-1)
    normed = _ref_ln(tail, s.eps)
    out = lin("final_layer.linear", normed * (1 + scale) + shift)
    grid = torch.zeros(b, c, hh * p, ww * p, dtype=torch.float64)
    for idx in range(hh * ww):
        i, j = divmod(idx, ww)
        grid[:, :, i * p:(i + 1) * p, j * p:(j + 1) * p] = out[:, idx].reshape(b, c, p, p)
    return grid


def case_parity() -> None:
    torch.manual_seed(0)
    shape = tiny_shape()
    model = flux.build(shape).double().eval()
    # 归一化层的 scale 不能是「全 1 默认值」⇒ 否则 qk-norm 那条路径与「不归一化」不可辨 ✓
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.query_norm.scale") or name.endswith("norm.key_norm.scale"):
                param.copy_(1.0 + 0.1 * torch.randn_like(param))
    lat = torch.randn(2, shape.channels, 6, 4, dtype=torch.float64)
    ctx = torch.randn(2, 5, shape.context_dim, dtype=torch.float64)
    pooled = torch.randn(2, shape.vec_dim, dtype=torch.float64)
    sigma = torch.tensor([0.73, 0.21], dtype=torch.float64)
    with torch.no_grad():
        got = model(lat, sigma, ctx, vector=pooled, guidance=3.5)
        want = _ref_forward(model, lat, sigma, ctx, pooled, 3.5)
    rel = float((got - want).abs().max() / want.abs().max())
    # 阈值 1e-7：这不是「容忍误差」，而是**照实标出这一族的两处 fp32**——``rope_freqs`` 出的
    # 旋转矩阵是 fp32（位置号可到几千，频率要 fp64 算、结果按 ComfyUI 口径降 fp32 ✓），
    # 时间/guidance 的正弦角度也在 fp32 里算 ✓。纯 fp64 参照因此有一个 ≈2.1e-9 的**下限** ✓
    # （实测把 RMSNorm 的 fp32 工作精度也去掉后仍是 2.1e-9 ⇒ 残差全部来自那两处 fp32 ✓）。
    # 判据强度：仍比 bf16 的分辨率（2⁻⁸ ≈ 4e-3）紧 4 个数量级；⑦ 的反验里两流换位 ⇒ 1.1e-1 ✓。
    check("③ 前向 == 独立双精度参照（rel < 1e-7，实测 floor ≈ 2e-9）",
          rel < 1e-7, f"rel={rel:.3e}")


# ---------------------------------------------------------------- ④ 拼接顺序 / 掩码


def case_order_and_mask() -> None:
    shape = tiny_shape()
    model = flux.build(shape).eval()
    bias = flux._attn_mask(torch.tensor([[True, False, True]]), 3, 7, torch.float32)
    row = bias[0, 0, 0]
    live = (row[:3] == 0).nonzero().reshape(-1).tolist()
    # 图像那 7 列**本来就该全 0**（不挡）⇒ 判据必须只看前 n_txt 列，否则「保留 0/2」会被
    # 7 个恒 0 的图像列稀释成 [0,2,3…9]，看着像拼反了（这是判据的坑，不是实现的 ✗）。
    tail_open = bool((row[3:] == 0).all())
    check("④ 掩码非零列恰好落在前 n_txt 列（txt 在前）", live == [0, 2] and tail_open,
          f"文本段保留位置 {live}、图像段全开 {tail_open}"
          f"（若拼成 img 在前 ⇒ 保留位会落到 3、5 上）")

    torch.manual_seed(1)
    lat = torch.randn(1, shape.channels, 4, 4)
    ctx = torch.randn(1, 4, shape.context_dim)
    keep = torch.tensor([[True, True, False, False]])
    with torch.no_grad():
        a = model(lat, torch.tensor([0.6]), ctx, vector=torch.zeros(1, shape.vec_dim),
                  attention_mask=keep)
        ctx2 = ctx.clone()
        ctx2[:, 2:] = 9.0   # 只改被挡掉的 pad 位
        b = model(lat, torch.tensor([0.6]), ctx2, vector=torch.zeros(1, shape.vec_dim),
                  attention_mask=keep)
        c = model(lat, torch.tensor([0.6]), ctx2, vector=torch.zeros(1, shape.vec_dim))
    check("④ 挡住文本 pad 位后，pad 内容不再影响输出",
          float((a - b).abs().max()) < 1e-6, f"差 {float((a - b).abs().max()):.3e}")
    check("④ 不挡掩码 ⇒ 同一改动**确实**会影响输出（掩码真的生效，不是自己骗自己）",
          float((a - c).abs().max()) > 1e-3, f"差 {float((a - c).abs().max()):.3e}")


# ---------------------------------------------------------------- ⑤ 三路 vec


def case_conditioning_paths() -> None:
    torch.manual_seed(2)
    shape = tiny_shape()
    model = flux.build(shape).eval()
    lat = torch.randn(1, shape.channels, 4, 4)
    ctx = torch.randn(1, 4, shape.context_dim)
    with torch.no_grad():
        g35 = model(lat, torch.tensor([0.5]), ctx, vector=torch.zeros(1, shape.vec_dim),
                    guidance=3.5)
        g10 = model(lat, torch.tensor([0.5]), ctx, vector=torch.zeros(1, shape.vec_dim),
                    guidance=1.0)
        v1 = model(lat, torch.tensor([0.5]), ctx, vector=torch.ones(1, shape.vec_dim),
                   guidance=3.5)
        d0 = model(lat, torch.tensor([0.5]), ctx, vector=torch.zeros(1, shape.vec_dim))
        dg = model(lat, torch.tensor([0.5]), ctx, vector=torch.zeros(1, shape.vec_dim),
                   guidance=shape.guidance_default)
    check("⑤ guidance 变化 ⇒ 输出变化", float((g35 - g10).abs().max()) > 1e-4,
          f"差 {float((g35 - g10).abs().max()):.3e}")
    check("⑤ pooled 向量变化 ⇒ 输出变化", float((g35 - v1).abs().max()) > 1e-4,
          f"差 {float((g35 - v1).abs().max()):.3e}")
    check("⑤ 不给 guidance ⇒ 等价于默认值（诚实降级，不静默换别的）",
          torch.allclose(d0, dg, atol=1e-6), f"差 {float((d0 - dg).abs().max()):.3e}")

    # 注入口：挂在模块上的默认条件必须和**同样取值**的显式传参等价 ✓
    # ⚠️ 对照必须用 (ones, guidance=1.0) ✓✗ —— 拿 g10（zeros 向量）比会差出 pooled 那一路的
    #    量级（≈8e-2），看着像注入口漏了，其实是判据自己配错了对照组 ✗。
    model.default_vector = torch.ones(1, shape.vec_dim)
    model.default_guidance = 1.0
    with torch.no_grad():
        via_default = model(lat, torch.tensor([0.5]), ctx)
        explicit = model(lat, torch.tensor([0.5]), ctx, vector=torch.ones(1, shape.vec_dim),
                         guidance=1.0)
    check("⑤ attach 注入（default_vector/guidance）== 显式传参",
          float((via_default - explicit).abs().max()) < 1e-6,
          f"差 {float((via_default - explicit).abs().max()):.3e}")


# ---------------------------------------------------------------- ⑥ 口径


def case_scales() -> None:
    """``1000·σ`` 且**不翻转**；位置号**不居中**。"""
    emb0 = generic.timestep_embedding(torch.tensor([0.0]), 8)
    emb1 = generic.timestep_embedding(torch.tensor([1.0]), 8)
    ref0 = _ref_timestep(torch.tensor([0.0]), 8)
    ref1 = _ref_timestep(torch.tensor([1.0]), 8)
    check("⑥ 时间嵌入 == 1000·σ 的正弦（与参照逐值一致）",
          torch.allclose(emb0.double(), ref0, atol=1e-6)
          and torch.allclose(emb1.double(), ref1, atol=1e-6))
    flipped = generic.timestep_embedding(torch.tensor([1.0]), 8)  # 若翻转，σ=0 会等于 σ=1 的嵌入
    check("⑥ σ 不翻转（embed(0) ≠ embed(1)）",
          float((emb0 - emb1).abs().max()) > 1e-2,
          f"差 {float((emb0 - flipped).abs().max()):.3e}")

    ids1 = flux._latent_ids(1, 1, torch.device("cpu"))
    check("⑥ 1×1 latent 的 h/w 号都是 0（不居中 ⇒ 旋转恒等）",
          ids1.reshape(3).tolist() == [0.0, 0.0, 0.0], f"{ids1.reshape(3).tolist()}")
    ids2 = flux._latent_ids(2, 1, torch.device("cpu")).reshape(2, 3)
    check("⑥ 2×1 latent 的 h 号是 0,1（**不居中** ⇒ 不是 −0.5/+0.5）",
          ids2[:, 1].tolist() == [0.0, 1.0], f"{ids2[:, 1].tolist()}")

    # ⚠️ ``1000·σ`` 曾被**硬写在前向里**：``timestep_embedding`` 的 `time_factor` 默认值恰好也是
    #    1000 ⇒ 前向漏传 `time_factor=s.t_scale` 时行为一模一样、`FluxShape.t_scale` 却成了
    #    「改它不生效」的摆设 ✗（改默认值才生效 ✗✗）。这条判据把「字段是活的」钉死。
    base = flux.build(tiny_shape()).double().eval()
    scaled = flux.build(dataclasses.replace(tiny_shape(), t_scale=1.0)).double().eval()
    scaled.load_state_dict(base.state_dict())
    lat = torch.randn(1, 8, 4, 4, dtype=torch.float64)
    ctx = torch.randn(1, 3, 12, dtype=torch.float64)
    sig = torch.tensor([0.5], dtype=torch.float64)
    zero = torch.zeros(1, 6, dtype=torch.float64)
    with torch.no_grad():
        full = base(lat, sig, ctx, vector=zero)
        tenth = scaled(lat, sig, ctx, vector=zero)
    gap = float((full - tenth).abs().max())
    check("⑥ ``FluxShape.t_scale`` 真被前向读（改 1000→1 必须改输出）", gap > 1e-3,
          f"差 {gap:.3e}")


# ---------------------------------------------------------------- ⑦ 严格性


def case_strictness() -> None:
    shape = tiny_shape()
    model = flux.build(shape).eval()
    good = torch.randn(1, shape.channels, 4, 4)
    cases = [
        ("context 宽错", good, torch.randn(1, 4, shape.context_dim + 1), None),
        ("latent 通道错", torch.randn(1, shape.channels + 4, 4, 4),
         torch.randn(1, 4, shape.context_dim), None),
        ("latent 尺寸非 patch 倍数", torch.randn(1, shape.channels, 5, 4),
         torch.randn(1, 4, shape.context_dim), None),
        ("vector 太窄", good, torch.randn(1, 4, shape.context_dim),
         torch.randn(1, shape.vec_dim - 2)),
    ]
    for label, lat, ctx, vec in cases:
        try:
            model(lat, torch.tensor([0.5]), ctx, vector=vec)
        except Exception as err:  # noqa: BLE001
            check(f"⑦ {label} ⇒ 具名抛（{type(err).__name__}）", True, str(err)[:70])
        else:
            check(f"⑦ {label} ⇒ 具名抛", False, "静默通过了 ✗")

    # ⚠️ 必须是 ``to(bfloat16)`` 而不是 ``half()`` ✗：``half()`` 给的是 float16 ✓✗ ——
    #    真权重是 bf16，用 fp16 判「bf16 进 bf16 出」等于没判（而且 fp16 下 attention 更容易炸）。
    bf = model.to(torch.bfloat16)
    with torch.no_grad():
        out = bf(good.to(torch.bfloat16), torch.tensor([0.5]),
                 torch.randn(1, 4, shape.context_dim, dtype=torch.bfloat16),
                 vector=torch.randn(1, shape.vec_dim, dtype=torch.bfloat16))
    check("⑦ bf16 进 ⇒ bf16 出（RoPE 后不被悄悄升 fp32，SDPA 不炸）",
          out.dtype == torch.bfloat16 and torch.isfinite(out).all(), f"{out.dtype}")

    # 反向：把 (txt, img) 拼接顺序真改坏 ⇒ ③ 的结论必须随之崩（否则那条对拍是空转）
    orig = flux._DoubleBlock.forward
    try:
        def broken(self, img, txt, vec, freqs, mask=None):  # noqa: ANN001
            out_img, out_txt = orig(self, img, txt, vec, freqs, mask)
            return out_txt, out_img
        flux._DoubleBlock.forward = broken  # type: ignore[assignment]
        torch.manual_seed(0)
        m2 = flux.build(shape).double().eval()
        with torch.no_grad():
            for name, p in m2.named_parameters():
                if "norm.query_norm.scale" in name or "norm.key_norm.scale" in name:
                    p.copy_(1.0 + 0.1 * torch.randn_like(p))
        lat = torch.randn(1, shape.channels, 4, 4, dtype=torch.float64)
        ctx = torch.randn(1, 3, shape.context_dim, dtype=torch.float64)
        sig = torch.tensor([0.5], dtype=torch.float64)
        with torch.no_grad():
            bad = m2(lat, sig, ctx, vector=torch.zeros(1, shape.vec_dim, dtype=torch.float64))
            ref = _ref_forward(m2, lat, sig, ctx,
                               torch.zeros(1, shape.vec_dim, dtype=torch.float64), None)
        rel = float((bad - ref).abs().max() / ref.abs().max())
        check("⑦ 反验：把文本/图像两流换位 ⇒ ③ 的对拍立刻报（证明它不是空转）",
              rel > 1e-6, f"rel={rel:.3e}")
    finally:
        flux._DoubleBlock.forward = orig  # type: ignore[assignment]


def case_real_load() -> None:
    """⑧ **真权重整份装载**（默认跳过：22 GiB 驻留 + 8 秒读盘 ⇒ 不能进 ``run_all``）。

    开法：``ENGINE_FLUX_REAL=1 .venv/Scripts/python.exe tests/engine_dit_flux_test.py`` ✓

    ⚠️ 这条跟 ① 是**两件不同的事** ✗：① 只比「键名与形状」，② 只在一份小配置上闭环；
    只有真文件走一遍 ``dit_build`` 才能证明 —— **装配表那一行接对了、meta 建壳能物化、
    780 个权重真的落了盘内张量、前向不因真实尺寸而炸** ✓。2026-11-01 实测：
    780/780（missing 0 / unexpected 0 / shapeMismatch 0）、驻留 **22.17 GiB**、
    CPU bf16 前向 1×16×32×32 一次 **1.5 秒**、guidance 3.5→7.0 输出差 **4.480** ✓
    （``generator(20261101)`` 固定 ⇒ 可复现；早先记的 2.25 是裸随机数的一次观测 ✗）。
    """
    if os.environ.get("ENGINE_FLUX_REAL") != "1" or not DEV.exists():
        print(f"  —— ⑧ 真权重装载跳过（ENGINE_FLUX_REAL=1 且盘上有 {DEV.name} 才跑）")
        return

    from app.services.engine import dit_build, dit_family, weights
    from app.services.engine import safetensors as st

    info = st.inspect(DEV)
    spec = dit_family.detect(info)
    check("⑧′ 真文件被判成 flux", spec.key == "flux", f"{spec.key}")

    module, cfg = dit_build.build("flux", info, {}, dtype=torch.float32, path=DEV)
    # ⚠️ 必须**先降 bf16 再装**：fp32 是 44.34 GiB，装到一半撞内存上限时崩的不是装载器
    #    而是别的进程 ✗（与设备策略一致：bf16 权重就该 bf16 驻留）。
    module = module.to(torch.bfloat16)
    report = weights.load_module_weights(module, DEV)
    ok = (report.loadedKeys == report.sourceKeys == 780 and not report.missing
          and not report.unexpected and not report.shapeMismatch and not report.error)
    check("⑧″ 780 键整份装载、无缺失/多余/形状不符", ok,
          f"{report.loadedKeys}/{report.sourceKeys} missing={len(report.missing)} "
          f"unexpected={len(report.unexpected)} mismatch={len(report.shapeMismatch)} "
          f"error={report.error!r}")

    gib = sum(p.numel() * p.element_size() for p in module.parameters()) / 2 ** 30
    check("⑧‴ bf16 驻留 ≈ 22.17 GiB（与参数 11.90 G 相符）", 21.9 < gib < 22.5, f"{gib:.2f} GiB")

    # ⚠️ 喂进去的噪声**必须固定 seed** ✗：原先裸 `torch.randn` ⇒ 「guidance 3.5→7.0 输出差 2.25」
    #    这句是**一次观测值**（复跑实测 5.398 ⇒ 记为 2.25 已失真 ✗）。判据本身只看 >1e-3 ✓，
    #    但写进文档的数字必须是能复现的 ⇒ 用局部 generator（不动全局 RNG ✓）。
    gen = torch.Generator().manual_seed(20261101)
    x = torch.randn(1, cfg["channels"] if "channels" in cfg else 16, 32, 32,
                    dtype=torch.bfloat16, generator=gen)
    ctx = torch.randn(1, 12, 4096, dtype=torch.bfloat16, generator=gen)
    with torch.no_grad():
        out = module(x, torch.tensor([0.7]), ctx)
        out7 = module(x, torch.tensor([0.7]), ctx, guidance=7.0)
    check("⑧⁗ 真尺寸前向出 (1,16,32,32) bf16 且有限",
          tuple(out.shape) == (1, 16, 32, 32) and out.dtype == torch.bfloat16
          and bool(torch.isfinite(out).all()), f"{tuple(out.shape)} {out.dtype}")
    delta = float((out7.float() - out.float()).abs().max())
    check("⑧⁵ 真权重上 guidance 通路仍生效（3.5→7.0 改变输出）", delta > 1e-3, f"Δ={delta:.3f}")

    del module, out, out7  # 立刻还内存（22 GiB 不释放会让后续 case 直接 OOM）
    import gc
    gc.collect()


def case_default_axes() -> None:
    """⑨ ``default_axes`` 必须对**任何** 4 的倍数都给三个**非零偶数**、且和 = head_dim。

    ⚠️ 这条是补上去的：原实现按 ``hd % 16 == 0`` 走 1:3.5:3.5 分支，``head_dim=16`` 算出
    ``(2, 7, 7)`` —— **建得成、跑不动**，``rope_freqs`` 当场抛「轴 1 的维度 7 必须是偶数」，
    而报错点落在 axes 上，读起来像调用方传错轴 ✗。自检里补这条才叫真正守住 ✓。
    """
    from app.services.engine.dit_generic import rope_freqs
    from app.services.engine.dit_flux import default_axes

    bad = []
    for hd in range(6, 130, 2):
        ax = default_axes(hd)
        if sum(ax) != hd or any(a <= 0 or a % 2 for a in ax):
            bad.append((hd, ax))
    check("⑨ 任意偶数 head_dim(≥6) ⇒ 三个非零偶数且和为 head_dim", not bad, f"越界 {bad}")

    named = {128: (16, 56, 56), 64: (8, 28, 28), 32: (4, 14, 14), 16: (4, 8, 4), 8: (2, 4, 2),
             12: (4, 4, 4), 10: (2, 4, 4)}
    wrong = {hd: default_axes(hd) for hd, want in named.items() if default_axes(hd) != want}
    check("⑨′ 关键档位与文档一致（32 的倍数走 1:3.5:3.5，8 的倍数走 ¼½¼，余者保底）",
          not wrong, f"不符 {wrong}")

    # 真跑一遍 RoPE：只看「不抛」，形状由 rope_freqs 自己保证
    try:
        ids = torch.zeros(1, 4, 3)
        f = rope_freqs(ids, default_axes(16), 10000.0)
        raised = ""
    except Exception as exc:  # noqa: BLE001
        raised = f"{type(exc).__name__}: {exc}"
    check("⑨″ head_dim=16 的三轴能真的建出 RoPE（原实现这里抛 ShapeError）",
          not raised, raised or f"ok {tuple(f.shape)}")


def case_attach_readout() -> None:
    """⑩ ``TorchBackend.attach_flux_conditioning`` 的回执必须报**这一拍真在生效**的值。

    ⚠️ 挂载是**写进模型属性**的 ⇒ 第二次只传 vector 时，上一次挂的 guidance **还在生效** ✗。
    原实现的 else 分支一律报 ``guidance_default``、vector 一律报「零向量」⇒ 回执与实际跑的东西
    对不上（最坏：以为退回默认 3.5，实际还在 7.0）✓✗。用 ``__new__`` 轻量替身测，不动显存。
    """
    from app.services.engine.torch_backend import TorchBackend

    shape = tiny_shape()
    model = flux.build(shape).eval()
    be = TorchBackend.__new__(TorchBackend)
    be._model = model
    be._ditFamily = "flux"
    be._gate = lambda: None
    be._torch = (lambda: torch)

    vec = torch.zeros(shape.vec_dim + 2)          # 故意多两维：既要能取前 vec_dim，也要回执如实
    first = be.attach_flux_conditioning(vector=vec, guidance=7.0)
    check("⑩ 第一次挂载：guidance 回执 = 传入值",
          abs(float(first["guidance"]) - 7.0) < 1e-9, str(first["guidance"]))

    second = be.attach_flux_conditioning()        # 什么都不给 ⇒ 沿用上次挂的，**不回落默认**
    live_ok = (abs(float(second["guidance"]) - 7.0) < 1e-9
               and abs(float(model.default_guidance) - 7.0) < 1e-9)
    check("⑩′ 第二次空调用：回执报**仍在生效**的 7.0（不是默认 3.5，也不是零向量）",
          live_ok, f"回执 {second['guidance']} / 模型 {model.default_guidance}")
    check("⑩″ 回执 vector 栏说清「沿用上次」而不是「零向量」（实际仍在生效）",
          "沿用" in str(second["vector"]), str(second["vector"]))


def main() -> int:
    print("== Flux.1 DiT 骨架自检 ==")
    for case in (case_real_header, case_shape_roundtrip, case_parity,
                 case_order_and_mask, case_conditioning_paths, case_scales,
                 case_strictness, case_default_axes, case_attach_readout, case_real_load):
        try:
            case()
        except Exception:  # noqa: BLE001
            check(f"{case.__name__} 未跑完 ✗", False, traceback.format_exc().splitlines()[-1])
    failed = [name for name, ok, _ in results if not ok]
    print(f"SUMMARY: {len(results) - len(failed)}/{len(results)} passed"
          + (f" ✗ 失败：{failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
